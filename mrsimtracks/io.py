import base64
import hashlib
import os
import re
import xml.etree.ElementTree as ET
import zlib

from collections.abc import Iterable
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyvista as pv

from tqdm.auto import tqdm
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator

from .sampler import (
    _TetSampler,
    _condition_mesh,
    resolve_float_dtype,
)


def _read_vtu(filepath, array_names, pbar):
    """Read a .vtu while loading only the requested point arrays.

    The pulsatile files store pressure_NNNNN alongside velocity_NNNNN for every
    timestep; tracking never uses pressure, so skipping it cuts both read time
    (~40%) and peak memory (~25%) -- important when each worker reloads the file.
    """
    if isinstance(array_names, str):
        array_names = [array_names]
    array_names = set(array_names)
    reader = pv.get_reader(filepath)
    reader.disable_all_point_arrays()
    missing = array_names.difference(reader.point_array_names)
    if missing:
        raise ValueError(
            f"point-data arrays {sorted(missing)!r} not found in {filepath}"
        )
    for name in array_names:
        reader.enable_point_array(name)
    if pbar:
        reader.show_progress()
    return reader.read()


@dataclass(frozen=True)
class MeshTopology:
    """One unique unstructured-grid topology."""

    cells: np.ndarray
    cell_types: np.ndarray

    @classmethod
    def from_mesh(cls, mesh):
        return cls(
            np.ascontiguousarray(mesh.cells).copy(),
            np.ascontiguousarray(mesh.celltypes).copy(),
        )

    def matches(self, mesh):
        return (
            np.array_equal(self.cells, np.asarray(mesh.cells))
            and np.array_equal(self.cell_types, np.asarray(mesh.celltypes))
        )


@dataclass(frozen=True)
class MeshFieldSeries:
    """Source-independent mesh coordinates, topologies, and point fields.

    Frame-to-storage index arrays make the common cases cheap: a static flow has
    one topology and one coordinate array, while a moving mesh has one topology
    and multiple coordinate arrays. Topology-changing data can retain multiple
    topologies without changing the representation.
    """

    times: np.ndarray
    topologies: tuple
    topology_ids: np.ndarray
    coordinates: tuple
    coordinate_ids: np.ndarray
    point_fields: dict

    def topology(self, frame):
        return self.topologies[int(self.topology_ids[frame])]

    def points(self, frame):
        return self.coordinates[int(self.coordinate_ids[frame])]

    def field(self, name, frame):
        return self.point_fields[name][frame]

    def mesh(self, frame):
        topology = self.topology(frame)
        return pv.UnstructuredGrid(
            topology.cells,
            topology.cell_types,
            self.points(frame),
            deep=False,
        )

    @property
    def geometry_mode(self):
        if len(self.topologies) > 1:
            return "changing_topology"
        if len(self.coordinates) > 1:
            return "moving"
        return "static"

    @property
    def bounds(self):
        lower = np.min([points.min(axis=0) for points in self.coordinates], axis=0)
        upper = np.max([points.max(axis=0) for points in self.coordinates], axis=0)
        return tuple(np.column_stack((lower, upper)).ravel())


def _center_mesh_frames(frames):
    """Translate geometry frames so the first frame's bounds center is zero."""
    initial = np.asarray(frames[0], dtype=np.float64)
    center = 0.5 * (initial.min(axis=0) + initial.max(axis=0))
    shift = -center
    translated = tuple(
        np.ascontiguousarray(
            np.asarray(points) + shift,
            dtype=np.asarray(points).dtype,
        )
        for points in frames
    )
    return translated, np.ascontiguousarray(shift)


def _center_mesh_data(data):
    coordinates, shift = _center_mesh_frames(data.coordinates)
    return replace(data, coordinates=coordinates), shift


@dataclass(frozen=True)
class _ArraySpec:
    association: str
    name: str
    vtk_type: str
    n_components: int
    format: str
    offset: int | None
    text: str | bytes | None
    n_tuples: int | None


@dataclass(frozen=True)
class _VTUMetadata:
    path: Path
    n_points: int
    n_cells: int
    byte_order: str
    header_type: str
    compressor: str | None
    appended_encoding: str | None
    appended_position: int | None
    arrays: tuple

    def arrays_in(self, association):
        return tuple(a for a in self.arrays if a.association == association)

    def array(self, association, name):
        for array in self.arrays:
            if array.association == association and array.name == name:
                return array
        raise KeyError((association, name))

    def next_offset(self, array):
        later = [
            a.offset
            for a in self.arrays
            if a.offset is not None and a.offset > array.offset
        ]
        return min(later) if later else None


class _UnsupportedFastPath(Exception):
    pass


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _child(element, name):
    return next((item for item in element if _tag(item) == name), None)


def _xml_attributes(source):
    attributes = {}
    pattern = rb"([A-Za-z_:][A-Za-z0-9_.:-]*)\s*=\s*(['\"])(.*?)\2"
    for match in re.finditer(pattern, source, flags=re.DOTALL):
        attributes[match.group(1).decode()] = match.group(3).decode()
    return attributes


def _inline_vtu_metadata(path, source):
    """Fast metadata scan for inline base64/ascii VTU files."""
    root_match = re.search(rb"<VTKFile\b([^>]*)>", source)
    piece_match = re.search(rb"<Piece\b([^>]*)>", source)
    if root_match is None or piece_match is None:
        raise ValueError(f"invalid VTU XML in {path}")
    root = _xml_attributes(root_match.group(1))
    piece = _xml_attributes(piece_match.group(1))

    arrays = []
    for association, container_name in (
        ("field", b"FieldData"),
        ("point", b"PointData"),
        ("cell", b"CellData"),
        ("points", b"Points"),
        ("cells", b"Cells"),
    ):
        container_start = source.find(b"<" + container_name)
        if container_start < 0:
            continue
        content_start = source.find(b">", container_start) + 1
        content_stop = source.find(b"</" + container_name, content_start)
        cursor = content_start
        while True:
            array_start = source.find(b"<DataArray", cursor, content_stop)
            if array_start < 0:
                break
            tag_stop = source.find(b">", array_start, content_stop) + 1
            attributes = _xml_attributes(source[array_start:tag_stop])
            if source[tag_stop - 2:tag_stop] == b"/>":
                text = None
                cursor = tag_stop
            else:
                data_stop = source.find(b"</DataArray", tag_stop, content_stop)
                text = source[tag_stop:data_stop]
                cursor = source.find(b">", data_stop, content_stop) + 1
            arrays.append(
                _ArraySpec(
                    association=association,
                    name=attributes.get("Name", ""),
                    vtk_type=attributes.get("type", "Float32"),
                    n_components=int(attributes.get("NumberOfComponents", "1")),
                    format=attributes.get("format", "ascii").lower(),
                    offset=(
                        int(attributes["offset"])
                        if "offset" in attributes
                        else None
                    ),
                    text=text,
                    n_tuples=(
                        int(attributes["NumberOfTuples"])
                        if "NumberOfTuples" in attributes
                        else None
                    ),
                )
            )
    return _VTUMetadata(
        path=Path(path),
        n_points=int(piece["NumberOfPoints"]),
        n_cells=int(piece["NumberOfCells"]),
        byte_order=root.get("byte_order", "LittleEndian"),
        header_type=root.get("header_type", "UInt32"),
        compressor=root.get("compressor"),
        appended_encoding=None,
        appended_position=None,
        arrays=tuple(arrays),
    )


def _read_vtu_metadata(path):
    """Read VTU XML metadata without touching raw appended array data."""
    path = Path(path)
    with path.open("rb") as file:
        prefix = bytearray()
        marker = -1
        while marker < 0:
            chunk = file.read(64 * 1024)
            if not chunk:
                break
            search_start = max(0, len(prefix) - len(b"<AppendedData"))
            prefix.extend(chunk)
            marker = prefix.find(b"<AppendedData", search_start)

        appended_encoding = None
        appended_position = None
        if marker >= 0:
            while b">" not in prefix[marker:]:
                prefix.extend(file.read(64 * 1024))
            tag_stop = prefix.find(b">", marker) + 1
            appended_tag = ET.fromstring(
                bytes(prefix[marker:tag_stop]) + b"</AppendedData>"
            )
            appended_encoding = appended_tag.get("encoding", "raw").lower()
            while b"_" not in prefix[tag_stop:]:
                prefix.extend(file.read(64 * 1024))
            appended_position = tag_stop + prefix[tag_stop:].find(b"_") + 1
            root = ET.fromstring(bytes(prefix[:marker]) + b"</VTKFile>")
        else:
            return _inline_vtu_metadata(path, bytes(prefix))

    grids = [item for item in root.iter() if _tag(item) == "UnstructuredGrid"]
    pieces = [item for item in root.iter() if _tag(item) == "Piece"]
    if len(grids) != 1 or len(pieces) != 1:
        raise _UnsupportedFastPath("only single-piece UnstructuredGrid VTU is optimized")
    grid, piece = grids[0], pieces[0]

    associations = {}
    field_data = _child(grid, "FieldData")
    point_data = _child(piece, "PointData")
    cell_data = _child(piece, "CellData")
    points = _child(piece, "Points")
    cells = _child(piece, "Cells")
    for association, container in (
        ("field", field_data),
        ("point", point_data),
        ("cell", cell_data),
        ("points", points),
        ("cells", cells),
    ):
        if container is not None:
            for element in container:
                if _tag(element) == "DataArray":
                    associations[id(element)] = association

    arrays = []
    for element in root.iter():
        if _tag(element) != "DataArray" or id(element) not in associations:
            continue
        arrays.append(
            _ArraySpec(
                association=associations[id(element)],
                name=element.get("Name", ""),
                vtk_type=element.get("type", "Float32"),
                n_components=int(element.get("NumberOfComponents", "1")),
                format=element.get("format", "ascii").lower(),
                offset=(int(element.get("offset")) if element.get("offset") else None),
                text=element.text,
                n_tuples=(
                    int(element.get("NumberOfTuples"))
                    if element.get("NumberOfTuples")
                    else None
                ),
            )
        )

    return _VTUMetadata(
        path=path,
        n_points=int(piece.get("NumberOfPoints")),
        n_cells=int(piece.get("NumberOfCells")),
        byte_order=root.get("byte_order", "LittleEndian"),
        header_type=root.get("header_type", "UInt32"),
        compressor=root.get("compressor"),
        appended_encoding=appended_encoding,
        appended_position=appended_position,
        arrays=tuple(arrays),
    )


_VTK_DTYPES = {
    "Int8": "i1",
    "UInt8": "u1",
    "Int16": "i2",
    "UInt16": "u2",
    "Int32": "i4",
    "UInt32": "u4",
    "Int64": "i8",
    "UInt64": "u8",
    "Float32": "f4",
    "Float64": "f8",
}


def _endian(metadata):
    return "<" if metadata.byte_order == "LittleEndian" else ">"


def _header_dtype(metadata):
    suffix = {"UInt32": "u4", "UInt64": "u8"}.get(metadata.header_type)
    if suffix is None:
        raise _UnsupportedFastPath(f"unsupported VTU header type {metadata.header_type!r}")
    return np.dtype(_endian(metadata) + suffix)


def _decode_encoded_binary(encoded, metadata):
    encoded = b"".join(encoded.split())
    header_dtype = _header_dtype(metadata)
    header_size = header_dtype.itemsize
    first_chars = 4 * ((header_size + 2) // 3)

    if metadata.compressor is None:
        header = base64.b64decode(encoded[:first_chars])[:header_size]
        size = int(np.frombuffer(header, header_dtype)[0])
        return base64.b64decode(encoded[first_chars:])[:size]

    if metadata.compressor != "vtkZLibDataCompressor":
        raise _UnsupportedFastPath(
            f"unsupported VTU compressor {metadata.compressor!r}"
        )
    first = base64.b64decode(encoded[:first_chars])[:header_size]
    n_blocks = int(np.frombuffer(first, header_dtype)[0])
    header_bytes = header_size * (3 + n_blocks)
    header_chars = 4 * ((header_bytes + 2) // 3)
    header = base64.b64decode(encoded[:header_chars])[:header_bytes]
    compressed_sizes = np.frombuffer(header, header_dtype)[3:].astype(np.int64)
    compressed = base64.b64decode(encoded[header_chars:])
    output = []
    cursor = 0
    for size in compressed_sizes:
        stop = cursor + int(size)
        output.append(zlib.decompress(compressed[cursor:stop]))
        cursor = stop
    return b"".join(output)


def _read_raw_appended(file, metadata):
    header_dtype = _header_dtype(metadata)
    header_size = header_dtype.itemsize
    if metadata.compressor is None:
        size = int(np.frombuffer(file.read(header_size), header_dtype)[0])
        return file.read(size)
    if metadata.compressor != "vtkZLibDataCompressor":
        raise _UnsupportedFastPath(
            f"unsupported VTU compressor {metadata.compressor!r}"
        )
    header = np.frombuffer(file.read(3 * header_size), header_dtype)
    n_blocks = int(header[0])
    compressed_sizes = np.frombuffer(
        file.read(n_blocks * header_size), header_dtype
    ).astype(np.int64)
    compressed = file.read(int(compressed_sizes.sum()))
    output = []
    cursor = 0
    for size in compressed_sizes:
        stop = cursor + int(size)
        output.append(zlib.decompress(compressed[cursor:stop]))
        cursor = stop
    return b"".join(output)


def _read_array(metadata, array, n_tuples=None):
    """Decode one DataArray without loading the VTU mesh."""
    dtype_suffix = _VTK_DTYPES.get(array.vtk_type)
    if dtype_suffix is None:
        raise _UnsupportedFastPath(f"unsupported VTK array type {array.vtk_type!r}")
    dtype = np.dtype(_endian(metadata) + dtype_suffix)
    text = array.text or b""
    if isinstance(text, str):
        text = text.encode()

    if array.format == "ascii":
        values = np.fromstring(text.decode(), sep=" ", dtype=dtype)
    elif array.format == "binary":
        values = np.frombuffer(
            _decode_encoded_binary(text, metadata), dtype
        )
    elif array.format == "appended":
        if metadata.appended_position is None or array.offset is None:
            raise ValueError(f"invalid appended DataArray {array.name!r}")
        with metadata.path.open("rb") as file:
            file.seek(metadata.appended_position + array.offset)
            if metadata.appended_encoding == "raw":
                raw = _read_raw_appended(file, metadata)
            elif metadata.appended_encoding == "base64":
                next_offset = metadata.next_offset(array)
                encoded = (
                    file.read(next_offset - array.offset)
                    if next_offset is not None
                    else file.read().split(b"</AppendedData>", 1)[0]
                )
                raw = _decode_encoded_binary(encoded, metadata)
            else:
                raise _UnsupportedFastPath(
                    f"unsupported appended encoding {metadata.appended_encoding!r}"
                )
        values = np.frombuffer(raw, dtype)
    else:
        raise _UnsupportedFastPath(f"unsupported DataArray format {array.format!r}")

    if n_tuples is None:
        n_tuples = array.n_tuples
    if n_tuples is not None:
        expected = n_tuples * array.n_components
        if values.size != expected:
            raise ValueError(
                f"{metadata.path}: array {array.name!r} has {values.size} values; "
                f"expected {expected}"
            )
        shape = (n_tuples, array.n_components) if array.n_components > 1 else (n_tuples,)
        values = values.reshape(shape)
    return values


def _payload_signature(metadata, array):
    """Hash the stored payload exactly, without decoding/decompressing it."""
    digest = hashlib.sha256()
    digest.update(
        f"{array.vtk_type}:{array.n_components}:{array.format}:".encode()
    )
    if array.format in {"ascii", "binary"}:
        text = array.text or b""
        if isinstance(text, str):
            text = text.encode()
        digest.update(b"".join(text.split()))
        return digest.digest()
    if array.format != "appended" or array.offset is None:
        raise _UnsupportedFastPath(f"cannot fingerprint {array.format!r} DataArray")

    next_offset = metadata.next_offset(array)
    with metadata.path.open("rb") as file:
        file.seek(metadata.appended_position + array.offset)
        if next_offset is not None:
            remaining = next_offset - array.offset
            while remaining:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(f"truncated appended data in {metadata.path}")
                digest.update(chunk)
                remaining -= len(chunk)
        elif metadata.appended_encoding == "raw":
            digest.update(_read_raw_appended(file, metadata))
        else:
            digest.update(file.read().split(b"</AppendedData>", 1)[0].strip())
    return digest.digest()


def _geometry_signatures(metadata):
    points = metadata.arrays_in("points")
    cells = metadata.arrays_in("cells")
    if len(points) != 1 or len(cells) < 3:
        raise _UnsupportedFastPath("VTU geometry arrays are incomplete")
    cell_by_name = {array.name: array for array in cells}
    required = ("connectivity", "offsets", "types")
    if not all(name in cell_by_name for name in required):
        raise _UnsupportedFastPath("VTU cell arrays are incomplete")
    topology = (
        metadata.n_cells,
        *(_payload_signature(metadata, cell_by_name[name]) for name in required),
    )
    coordinates = (
        metadata.n_points,
        _payload_signature(metadata, points[0]),
    )
    return topology, coordinates


_TIME_INTERP = ("linear", "cubic")
_MESH_MODE_ALIASES = {
    "auto": "auto",
    "static": "static",
    "moving": "moving",
    "moving-node": "moving",
    "moving_node": "moving",
    "changing_topology": "changing_topology",
    "changing-topology": "changing_topology",
}


def resolve_time_interp(time_interp):
    """Validate the temporal interpolation mode (``"linear"`` or ``"cubic"``)."""
    if time_interp not in _TIME_INTERP:
        raise ValueError(
            f"time_interp must be one of {_TIME_INTERP}, got {time_interp!r}")
    return time_interp


def resolve_mesh_mode(mesh_mode):
    """Validate and normalize the mesh classification policy."""
    try:
        return _MESH_MODE_ALIASES[mesh_mode]
    except (KeyError, TypeError):
        options = ("auto", "static", "moving", "changing_topology")
        raise ValueError(
            f"mesh_mode must be one of {options}, got {mesh_mode!r}"
        ) from None


def _catmull_rom(p0, p1, p2, p3, s):
    """Uniform Catmull-Rom spline value at local parameter ``s`` in [0, 1].

    Interpolates the smooth segment between knots ``p1`` and ``p2`` using the
    neighbours ``p0`` and ``p3`` to estimate the endpoint tangents. Reproduces
    each knot exactly (s=0 -> p1, s=1 -> p2). ``s`` is a python float so an f32
    field stays f32.

    Uses scalar basis weights (not array-valued coefficients) so each frame is
    touched by a single scalar-times-array multiply -- only ~2x the linear blend
    rather than the many full-field temporaries a Horner form allocates.
    """
    s2 = s * s
    s3 = s2 * s
    w0 = 0.5 * (-s + 2.0 * s2 - s3)
    w1 = 0.5 * (2.0 - 5.0 * s2 + 3.0 * s3)
    w2 = 0.5 * (s + 4.0 * s2 - 3.0 * s3)
    w3 = 0.5 * (-s2 + s3)
    return w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3


def _periodic_distinct_count(get_frame, n_frames):
    """Number of distinct frames per period.

    A pulsatile series often stores the period's closing frame as a duplicate of
    the opening one; if so it is dropped from the periodic wrap so cubic
    interpolation across the cycle boundary doesn't see a repeated knot.
    """
    f0 = np.asarray(get_frame(0))
    fn = np.asarray(get_frame(n_frames - 1))
    if f0.shape != fn.shape:
        return n_frames
    scale = float(np.abs(f0).max()) or 1.0
    dup = np.allclose(f0, fn, rtol=1e-3, atol=1e-3 * scale)
    return n_frames - 1 if dup else n_frames


def _require_uniform_spacing(times_shift_s, mode):
    """Cubic interpolation assumes uniform frame spacing -- check it once."""
    if mode == "cubic":
        d = np.diff(np.asarray(times_shift_s, dtype=float))
        if d.size and not np.allclose(d, d[0], rtol=1e-6, atol=1e-12):
            raise ValueError(
                "time_interp='cubic' requires uniformly spaced time frames")


def _interp_time(times_shift_s, tmax, n_distinct, get_frame, time, mode,
                 tol=1e-3):
    """Interpolate the nodal field at ``time`` (periodic) with ``mode``.

    ``get_frame(i)`` returns the nodal velocity array for frame index ``i``.
    Linear reproduces the legacy two-frame blend exactly; cubic uses a uniform
    Catmull-Rom across four frames, wrapping neighbours periodically.
    """
    indices, weights = _interp_weights(
        times_shift_s, tmax, n_distinct, time, mode, tol=tol
    )
    if len(indices) == 1:
        return get_frame(indices[0])
    output = weights[0] * get_frame(indices[0])
    for index, weight in zip(indices[1:], weights[1:], strict=True):
        output = output + weight * get_frame(index)
    return output


def _interp_weights(times_shift_s, tmax, n_distinct, time, mode, tol=1e-3):
    """Frame indices and scalar weights for periodic temporal interpolation."""
    tw = time % tmax
    inext = int(np.argmax((times_shift_s - tw) > 0))
    iprev = inext - 1
    s = float(
        (tw - times_shift_s[iprev])
        / (times_shift_s[inext] - times_shift_s[iprev])
    )
    if s < tol:
        return (iprev,), (1.0,)
    if s > 1 - tol:
        return (inext,), (1.0,)
    if mode == "cubic":
        nd = n_distinct
        s2 = s * s
        s3 = s2 * s
        return (
            (iprev - 1) % nd,
            iprev % nd,
            inext % nd,
            (inext + 1) % nd,
        ), (
            0.5 * (-s + 2.0 * s2 - s3),
            0.5 * (2.0 - 5.0 * s2 + 3.0 * s3),
            0.5 * (s + 4.0 * s2 - 3.0 * s3),
            0.5 * (-s2 + s3),
        )
    return (iprev, inext), (1.0 - s, s)


def _resolve_point_array(metadata, requested, *, suffixed=False):
    names = [array.name for array in metadata.arrays_in("point")]
    if suffixed:
        pattern = re.compile(rf"^{re.escape(requested)}_(\d+)$", re.IGNORECASE)
        matches = []
        for name in names:
            match = pattern.match(name)
            if match:
                matches.append((int(match.group(1)), name))
        matches.sort()
        return matches

    exact = [name for name in names if name == requested]
    folded = [name for name in names if name.casefold() == requested.casefold()]
    matches = exact or folded
    if len(matches) != 1:
        available = ", ".join(repr(name) for name in names)
        raise ValueError(
            f"point-data array {requested!r} not found uniquely in {metadata.path}; "
            f"available arrays: {available}"
        )
    return matches[0]


def _parse_pvd(filepath):
    """Return ``(time, VTU path)`` entries sorted by time."""
    filepath = Path(filepath)
    root = ET.parse(filepath).getroot()
    entries = []
    for dataset in root.iter():
        if _tag(dataset) != "DataSet":
            continue
        filename = dataset.get("file")
        if filename is None:
            continue
        entries.append((float(dataset.get("timestep", "0")), filepath.parent / filename))
    entries.sort(key=lambda item: item[0])
    if not entries:
        raise ValueError(f"{filepath} does not contain any DataSet entries")
    return entries


def _filename_time(path):
    match = re.search(r"(\d+(?:\.\d+)?)$", Path(path).stem)
    if match is None:
        raise ValueError(
            f"cannot infer a timestep from {path}; use names ending in a number"
        )
    return float(match.group(1))


def _metadata_time(metadata):
    for array in metadata.arrays_in("field"):
        if array.name.casefold() == "timevalue":
            value = np.asarray(_read_array(metadata, array, n_tuples=1)).ravel()
            return float(value[0])
    return None


def _series_source(path, active_key):
    """Resolve a PVD, directory, or file list to sorted VTU frames."""
    if isinstance(path, (str, os.PathLike)):
        source = Path(path)
        if source.is_file() and source.suffix.lower() == ".pvd":
            entries = _parse_pvd(source)
            return entries, None
        if not source.is_dir():
            raise ValueError(
                f"unsupported flow file type: {source.suffix or source} "
                "(expected .vtu, .pvd, a directory, or a VTU file list)"
            )
        candidates = sorted(source.glob("*.vtu"))
    else:
        candidates = [Path(file) for file in path]

    selected = []
    for file in candidates:
        if file.suffix.lower() != ".vtu":
            continue
        metadata = _read_vtu_metadata(file)
        try:
            _resolve_point_array(metadata, active_key)
        except ValueError:
            continue
        time_value = _metadata_time(metadata)
        selected.append(
            (time_value if time_value is not None else _filename_time(file), file)
        )
    selected.sort(key=lambda item: item[0])
    if len(selected) < 2:
        raise ValueError("a VTU series must contain at least two flow frames")
    times = [item[0] for item in selected]
    if len(set(times)) != len(times):
        raise ValueError("VTU series timesteps must be unique")
    return selected, None


def _find_or_add_topology(topologies, mesh):
    for index, topology in enumerate(topologies):
        if topology.matches(mesh):
            return index
    topologies.append(MeshTopology.from_mesh(mesh))
    return len(topologies) - 1


def _find_or_add_coordinates(coordinates, points):
    points = np.ascontiguousarray(points)
    for index, existing in enumerate(coordinates):
        if np.array_equal(existing, points):
            return index
    coordinates.append(points.copy())
    return len(coordinates) - 1


def _load_vtu_series(entries, metadata, active_key, *, subsamp, dt, pbar,
                     dtype, conform_mesh, mesh_mode):
    if subsamp < 1:
        raise ValueError("subsamp must be >= 1")
    entries = entries[::subsamp]
    if metadata is not None:
        metadata = metadata[::subsamp]
    if len(entries) < 2:
        raise ValueError("flow input must contain at least two timesteps")

    raw_times = np.asarray([time for time, _ in entries])
    fields = []
    topologies = []
    coordinates = []
    topology_ids = []
    coordinate_ids = []
    topology_signatures = {}
    coordinate_signatures = {}
    canonical_key = None
    reference_topology = None
    reference_n_points = None
    midpoint = len(entries) // 2

    iterator = tqdm(entries, total=len(entries), disable=not pbar)
    for frame, (_, file) in enumerate(iterator):
        info = (
            _read_vtu_metadata(file)
            if metadata is None
            else metadata[frame]
        )
        key = _resolve_point_array(info, active_key)
        if canonical_key is None:
            canonical_key = key

        full_mesh = None
        if frame == 0:
            full_mesh = _read_vtu(file, [key], pbar=False)
            reference_topology = MeshTopology.from_mesh(full_mesh)
            if conform_mesh:
                full_mesh = _condition_mesh(full_mesh)
            topology_id = _find_or_add_topology(topologies, full_mesh)
            coordinate_id = _find_or_add_coordinates(coordinates, full_mesh.points)
            reference_n_points = full_mesh.n_points
            if mesh_mode == "auto":
                try:
                    topology_signature, coordinate_signature = (
                        _geometry_signatures(info)
                    )
                except _UnsupportedFastPath:
                    topology_signature = coordinate_signature = None
                if topology_signature is not None:
                    topology_signatures[topology_signature] = topology_id
                    coordinate_signatures[coordinate_signature] = coordinate_id
        elif mesh_mode in {"static", "moving"}:
            if info.n_points != reference_n_points:
                raise ValueError(
                    f"mesh_mode={mesh_mode!r} requires {reference_n_points} "
                    f"point values per frame, but {file} declares "
                    f"{info.n_points} points"
                )
            topology_id = 0
            if frame == midpoint:
                full_mesh = _read_vtu(file, [key], pbar=False)
                if not reference_topology.matches(full_mesh):
                    raise ValueError(
                        f"mesh_mode={mesh_mode!r} midpoint frame has different "
                        f"topology: {file}"
                    )
            if mesh_mode == "static":
                coordinate_id = 0
                if frame == midpoint:
                    if not np.array_equal(coordinates[0], full_mesh.points):
                        raise ValueError(
                            "mesh_mode='static' midpoint frame has different "
                            f"node locations: {file}"
                        )
            else:
                if full_mesh is not None:
                    points = np.asarray(full_mesh.points)
                else:
                    points_array = info.arrays_in("points")
                    if len(points_array) != 1:
                        raise ValueError(f"{file} does not contain one points array")
                    try:
                        points = _read_array(
                            info, points_array[0], n_tuples=reference_n_points
                        )
                    except _UnsupportedFastPath:
                        full_mesh = _read_vtu(file, [key], pbar=False)
                        points = np.asarray(full_mesh.points)
                coordinate_id = _find_or_add_coordinates(coordinates, points)
        elif mesh_mode == "changing_topology":
            full_mesh = _read_vtu(file, [key], pbar=False)
            if conform_mesh:
                full_mesh = _condition_mesh(full_mesh)
            topology_id = _find_or_add_topology(topologies, full_mesh)
            coordinate_id = _find_or_add_coordinates(coordinates, full_mesh.points)
        else:
            try:
                topology_signature, coordinate_signature = (
                    _geometry_signatures(info)
                )
            except _UnsupportedFastPath:
                topology_signature = coordinate_signature = None
            if (
                topology_signature is None
                or topology_signature not in topology_signatures
            ):
                full_mesh = _read_vtu(file, [key], pbar=False)
                if conform_mesh:
                    full_mesh = _condition_mesh(full_mesh)
                topology_id = _find_or_add_topology(topologies, full_mesh)
                coordinate_id = _find_or_add_coordinates(
                    coordinates, full_mesh.points
                )
                if topology_signature is not None:
                    topology_signatures[topology_signature] = topology_id
                    coordinate_signatures[coordinate_signature] = coordinate_id
            else:
                topology_id = topology_signatures[topology_signature]
                if coordinate_signature in coordinate_signatures:
                    coordinate_id = coordinate_signatures[coordinate_signature]
                else:
                    points_array = info.arrays_in("points")[0]
                    points = _read_array(
                        info, points_array, n_tuples=info.n_points
                    )
                    coordinate_id = _find_or_add_coordinates(coordinates, points)
                    coordinate_signatures[coordinate_signature] = coordinate_id

        if full_mesh is None:
            try:
                field = _read_array(
                    info, info.array("point", key), n_tuples=info.n_points
                )
            except _UnsupportedFastPath:
                full_mesh = _read_vtu(file, [key], pbar=False)
                if conform_mesh:
                    full_mesh = _condition_mesh(full_mesh)
                field = np.asarray(full_mesh.point_data[key])
        else:
            field = np.asarray(full_mesh.point_data[key])
        if field.ndim != 2 or field.shape[1] != 3:
            raise ValueError(f"active point-data array {key!r} must have three components")
        if mesh_mode in {"static", "moving"} and field.shape[0] != reference_n_points:
            raise ValueError(
                f"mesh_mode={mesh_mode!r} requires {reference_n_points} point "
                f"values per frame, but {file} field {key!r} has {field.shape[0]}"
            )
        fields.append(np.ascontiguousarray(field, dtype=dtype).copy())
        topology_ids.append(topology_id)
        coordinate_ids.append(coordinate_id)

    times_shift_s = raw_times - raw_times[0]
    if dt is not None:
        times_shift_s = times_shift_s * dt
    data = MeshFieldSeries(
        times=raw_times,
        topologies=tuple(topologies),
        topology_ids=np.asarray(topology_ids, dtype=np.int64),
        coordinates=tuple(coordinates),
        coordinate_ids=np.asarray(coordinate_ids, dtype=np.int64),
        point_fields={canonical_key: tuple(fields)},
    )
    return data, canonical_key, times_shift_s


def _load_single_vtu(path, active_key, *, subsamp, only_active_key, pbar, dtype,
                     conform_mesh):
    if subsamp < 1:
        raise ValueError("subsamp must be >= 1")
    metadata = _read_vtu_metadata(path)
    time_keys = _resolve_point_array(metadata, active_key, suffixed=True)[::subsamp]
    if len(time_keys) < 2:
        raise ValueError(
            f"{path} must contain at least two point-data arrays named "
            f"{active_key}_NNNNN"
        )
    names = [name for _, name in time_keys]
    mesh = (
        _read_vtu(path, names, pbar)
        if only_active_key
        else pv.read(path, progress_bar=pbar)
    )
    if conform_mesh:
        mesh = _condition_mesh(mesh)
    canonical_key = names[0].rsplit("_", 1)[0]
    fields = []
    for name in names:
        field = np.asarray(mesh.point_data[name])
        if field.ndim != 2 or field.shape[1] != 3:
            raise ValueError(f"active point-data array {name!r} must have three components")
        fields.append(np.ascontiguousarray(field, dtype=dtype).copy())
    times = np.asarray([time for time, _ in time_keys])
    data = MeshFieldSeries(
        times=times,
        topologies=(MeshTopology.from_mesh(mesh),),
        topology_ids=np.zeros(len(times), dtype=np.int64),
        coordinates=(np.ascontiguousarray(mesh.points).copy(),),
        coordinate_ids=np.zeros(len(times), dtype=np.int64),
        point_fields={canonical_key: tuple(fields)},
    )
    return data, canonical_key, (times - times[0]) / 1000.0


@dataclass
class _FrameRuntime:
    mesh: pv.UnstructuredGrid
    locator: vtkStaticCellLocator
    sampler: _TetSampler


class Flow:
    """A time-resolved field over static, moving, or changing mesh geometry."""

    def __init__(self, data, active_key, times_shift_s, *, dtype, time_interp,
                 origin_shift=None):
        self.data = data
        self.active_key = active_key
        self.dtype = np.dtype(dtype)
        self.origin_shift = np.zeros(3) if origin_shift is None else np.asarray(
            origin_shift, dtype=float
        )
        self.time_interp = resolve_time_interp(time_interp)
        self.times = np.asarray(data.times)
        self.times_shift_s = np.asarray(times_shift_s, dtype=float)
        self.tmax = float(self.times_shift_s[-1])
        if self.tmax <= 0:
            raise ValueError("flow timesteps must be strictly increasing")
        if np.any(np.diff(self.times_shift_s) <= 0):
            raise ValueError("flow timesteps must be strictly increasing")
        _require_uniform_spacing(self.times_shift_s, self.time_interp)
        self._n_distinct = _periodic_distinct_count(self._frame_vel, len(self.times))
        self.fields = list(data.point_fields[active_key])
        self.geometry_mode = data.geometry_mode
        self.bounds = data.bounds
        self._runtime_cache = OrderedDict()

        runtime = self._frame_runtime(0)
        self.active_mesh = runtime.mesh
        self.active_mesh.point_data[self.active_key] = self._frame_vel(0).copy()
        self.mesh = self.active_mesh
        self.locator = runtime.locator
        self._sampler = runtime.sampler

    def _frame_vel(self, index):
        return self.data.field(self.active_key, index)

    def _frame_runtime(self, index):
        key = (
            int(self.data.topology_ids[index]),
            int(self.data.coordinate_ids[index]),
        )
        runtime = self._runtime_cache.pop(key, None)
        if runtime is not None:
            self._runtime_cache[key] = runtime
            return runtime
        mesh = self.data.mesh(index)
        locator = vtkStaticCellLocator()
        locator.SetDataSet(mesh)
        locator.BuildLocator()
        runtime = _FrameRuntime(mesh, locator, _TetSampler(mesh, dtype=self.dtype))
        self._runtime_cache[key] = runtime
        if len(self._runtime_cache) > 4:
            self._runtime_cache.popitem(last=False)
        return runtime

    def _sample_frame(self, index, points_xyz, guess=None):
        runtime = self._frame_runtime(index)
        field = self._frame_vel(index)
        if runtime.sampler.ok:
            return runtime.sampler.sample(points_xyz, field, guess=guess)
        runtime.mesh.point_data[self.active_key] = field
        sampled = pv.PolyData(points_xyz).sample(
            runtime.mesh,
            locator=runtime.locator,
            pass_cell_data=False,
            pass_point_data=False,
            pass_field_data=False,
        )
        valid = np.asarray(sampled["vtkValidPointMask"]).astype(bool)
        velocity = np.asarray(sampled[self.active_key]).copy()
        velocity[~valid] = 0
        return velocity, valid, None

    def _sample_frame_weights(self, points_xyz, weights, guess=None):
        indices = np.flatnonzero(weights)
        if self.geometry_mode == "static":
            runtime = self._frame_runtime(0)
            if runtime.sampler.ok:
                field = np.zeros_like(self._frame_vel(0))
                for index in indices:
                    field += weights[index] * self._frame_vel(index)
                return runtime.sampler.sample(points_xyz, field, guess=guess)

        velocity = np.zeros((len(points_xyz), 3), dtype=self.dtype)
        valid = np.ones(len(points_xyz), dtype=bool)
        cells = None
        same_topology = len(set(self.data.topology_ids[indices])) == 1
        for index in indices:
            values, frame_valid, frame_cells = self._sample_frame(
                int(index), points_xyz, guess=guess if same_topology else None
            )
            velocity += weights[index] * values
            valid &= frame_valid
            if cells is None:
                cells = frame_cells
        velocity[~valid] = 0
        return velocity, valid, cells if same_topology else None

    def set_active_time(self, time):
        if self.geometry_mode == "static":
            self.active_mesh.point_data[self.active_key] = _interp_time(
                self.times_shift_s,
                self.tmax,
                self._n_distinct,
                self._frame_vel,
                time,
                self.time_interp,
            )
            return
        self.active_mesh = self.get_mesh(time)
        self.mesh = self.active_mesh

    def get_mesh(self, time):
        indices, weights = _interp_weights(
            self.times_shift_s,
            self.tmax,
            self._n_distinct,
            time,
            self.time_interp,
        )
        topology_ids = {int(self.data.topology_ids[index]) for index in indices}
        if len(topology_ids) == 1:
            topology = self.data.topologies[topology_ids.pop()]
            points = sum(
                weight * self.data.points(index)
                for index, weight in zip(indices, weights, strict=True)
            )
            field = sum(
                weight * self._frame_vel(index)
                for index, weight in zip(indices, weights, strict=True)
            )
            mesh = pv.UnstructuredGrid(
                topology.cells,
                topology.cell_types,
                np.ascontiguousarray(points),
                deep=False,
            )
            mesh.point_data[self.active_key] = np.ascontiguousarray(field)
            return mesh
        index = indices[int(np.argmax(np.abs(weights)))]
        mesh = self.data.mesh(index)
        mesh.point_data[self.active_key] = self._frame_vel(index)
        return mesh

    def sample(self, points, time):
        velocity, valid, _ = self.sample_v(np.asarray(points.points), time)
        output = points.copy()
        output.point_data[self.active_key] = velocity
        output.point_data["vtkValidPointMask"] = valid.astype(np.uint8)
        return output

    def sample_v(self, points_xyz, time, guess=None):
        points_xyz = np.ascontiguousarray(points_xyz, dtype=self.dtype)
        if self.geometry_mode == "static":
            self.set_active_time(time)
            if not self._sampler.ok:
                indices, weights = _interp_weights(
                    self.times_shift_s,
                    self.tmax,
                    self._n_distinct,
                    time,
                    self.time_interp,
                )
                frame_weights = np.zeros(len(self.times), dtype=float)
                frame_weights[list(indices)] = weights
                return self._sample_frame_weights(
                    points_xyz, frame_weights, guess=guess
                )
            velocity = np.asarray(self.active_mesh.point_data[self.active_key])
            return self._sampler.sample(points_xyz, velocity, guess=guess)

        indices, weights = _interp_weights(
            self.times_shift_s,
            self.tmax,
            self._n_distinct,
            time,
            self.time_interp,
        )
        frame_weights = np.zeros(len(self.times), dtype=float)
        frame_weights[list(indices)] = weights
        return self._sample_frame_weights(points_xyz, frame_weights, guess=guess)


def load_flow(
    path: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    active_key: str = "velocity",
    subsamp: int = 1,
    only_active_key: bool = True,
    pbar: bool = False,
    dt: float | None = None,
    precision: str = "f64",
    time_interp: str = "linear",
    conform_mesh: bool = True,
    mesh_mode: str = "auto",
    center_mesh: bool = False,
) -> Flow:
    """Load a time-resolved flow into one source-independent representation.

    Supported sources are a single ``.vtu`` containing ``name_NNNNN`` point
    fields, a ``.pvd`` collection, a directory of per-frame ``.vtu`` files, or
    an explicit iterable of ``.vtu`` paths. Separate-file series are checked
    frame by frame and classified as static, moving-node, or topology-changing
    unless ``mesh_mode`` declares the layout. Identical topology and coordinates
    are stored once; only changed arrays are decoded after the first full mesh
    load.

    Args:
        path: A VTU/PVD path, directory, or iterable of VTU paths.
        active_key: Three-component point-data field name. Matching is
            case-insensitive; a single VTU expects ``active_key_NNNNN`` arrays.
        subsamp: Keep every Nth time frame for every source layout.
        only_active_key: Skip unrelated point arrays in a single multi-field VTU.
        pbar: Show load progress.
        dt: Optional multiplier for PVD, directory, or file-list time labels.
        precision: Working field precision, ``"f64"`` or ``"f32"``.
        time_interp: ``"linear"`` or uniform-grid ``"cubic"`` interpolation.
        conform_mesh: Split supported non-tetrahedral cells and remove
            degenerate tetrahedra before building the fast sampler.
        mesh_mode: ``"auto"`` to classify every frame, ``"static"`` to reuse
            the first mesh while checking midpoint coordinates and topology,
            ``"moving"`` to reuse the first connectivity while loading
            coordinates per frame and checking midpoint topology, or
            ``"changing_topology"`` to load geometry per frame. The aliases
            ``"moving-node"`` and ``"moving_node"`` are accepted.
        center_mesh: Translate every stored mesh frame by the same vector so
            the initial frame's axis-aligned bounds are centered at the origin.
            The default is ``False``; point fields are unchanged.

    Returns:
        Flow: Unified flow object used by tracking, reseeding, and imaging.
    """
    dtype = resolve_float_dtype(precision)
    time_interp = resolve_time_interp(time_interp)
    mesh_mode = resolve_mesh_mode(mesh_mode)

    if isinstance(path, (str, os.PathLike)) and Path(path).suffix.lower() == ".vtu":
        if mesh_mode not in {"auto", "static"}:
            raise ValueError(
                "a single time-field VTU has one mesh; mesh_mode must be "
                "'auto' or 'static'"
            )
        data, key, times_shift_s = _load_single_vtu(
            path,
            active_key,
            subsamp=subsamp,
            only_active_key=only_active_key,
            pbar=pbar,
            dtype=dtype,
            conform_mesh=conform_mesh,
        )
    else:
        entries, metadata = _series_source(path, active_key)
        data, key, times_shift_s = _load_vtu_series(
            entries,
            metadata,
            active_key,
            subsamp=subsamp,
            dt=dt,
            pbar=pbar,
            dtype=dtype,
            conform_mesh=conform_mesh,
            mesh_mode=mesh_mode,
        )
    origin_shift = np.zeros(3)
    if center_mesh:
        data, origin_shift = _center_mesh_data(data)
    return Flow(
        data,
        key,
        times_shift_s,
        dtype=dtype,
        time_interp=time_interp,
        origin_shift=origin_shift,
    )
