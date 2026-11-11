# PDB molecule import

`PDBLoader.load` reads a local file, UTF-8 data, or a string. The result
contains a `PointCloud` with linear CPK vertex colors, an optional
`LineGeometry` with one segment per resolved bond, and structured atom and
bond records. Coordinates retain the authored positions in angstroms.
ATOM and HETATM records retain their serial, name, element, residue,
chain, residue sequence, insertion code, alternate location, occupancy,
and temperature factor. The loader uses the declared element or infers it
from the aligned atom name. Unknown elements use neutral gray. The
original CPK palette is from the Three.js r187 PDB loader.
[Three.js r187 PDBLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/PDBLoader.js)
[Three.js MIT notice](../../LICENSES/Three-CurveExtras.txt)
[wwPDB coordinate records](https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html)

The loader selects the first explicit MODEL and accepts CONECT records
inside that model or outside the model blocks. It resolves up to four
fixed-width targets per CONECT record by atom serial number. Duplicate
and reciprocal connections collapse to one bond in the first recorded
direction. Self-connections, zero placeholders, and unresolved references
are skipped. Connectivity comes from explicit records, with one segment
per atom pair. Bond-order interpretation, inferred bonds, later models,
hybrid-36 serial numbers, and mmCIF are outside this reader's scope.
The loader checks fixed columns, finite positions and numeric metadata,
unique decimal atom serials, model boundaries, and invalid connection
fields. It limits input to 64 MiB and distinct CONECT pairs to 1,000,000.
[wwPDB connectivity records](https://www.wwpdb.org/documentation/file-format-content/format33/sect10.html)

## Verification

All 18 pinned Three.js r187 molecule fixtures matched the pinned
JavaScript loader on macOS and iPad Simulator. The 1,253 atom positions
and 1,888 bond segments matched bit for bit in source order. Element
symbols matched, and the maximum absolute difference in linear RGB was
0.00000018. The focused fixture run passed 33 test declarations on each
target. A Metal render check samples both imported atom colors, their
CONECT line, and the transparent background in the same frame.
[Three.js r187 PDB fixtures](https://github.com/mrdoob/three.js/tree/r187/examples/models/pdb)

The Xcode 27.2 MCP timed out resolving the package destination.
Verification used `xcodebuild` with internal DerivedData and Metal API
and shader validation enabled.
