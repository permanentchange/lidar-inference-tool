"""Read the same unmodified point features used by the native demo loaders."""
from pathlib import Path
import numpy as np


def load_points(path, features=5):
    path = Path(path)
    if path.suffix == '.npy':
        return np.load(path)
    if path.suffix == '.bin':
        return np.fromfile(path, dtype=np.float32).reshape(-1, features)
    header = {}
    with path.open('rb') as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('PCD DATA header missing: ' + str(path))
            fields = line.decode('ascii').split()
            if not fields or fields[0].startswith('#'):
                continue
            header[fields[0]] = fields[1:]
            if fields[0] == 'DATA':
                break
        names = header['FIELDS']
        if header['DATA'] == ['binary']:
            counts = header.get('COUNT', ['1'] * len(names))
            dtype = np.dtype([(n, {'F':'f','U':'u','I':'i'}[t] + size) if int(c)==1 else
                              (n, {'F':'f','U':'u','I':'i'}[t] + size, int(c))
                              for n,t,size,c in zip(names, header['TYPE'], header['SIZE'], counts)])
            points = np.fromfile(stream, dtype=dtype, count=int(header['POINTS'][0]))
            columns = {name: points[name] for name in names}
        elif header['DATA'] == ['ascii']:
            points = np.loadtxt(stream, dtype=np.float32, ndmin=2)
            columns = {name: points[:,i] for i,name in enumerate(names)}
        else:
            raise ValueError('Unsupported PCD encoding: ' + str(path))
    result = np.zeros((len(points), features), dtype=np.float32)
    for i,name in enumerate(('x','y','z')):
        result[:,i] = columns[name]
    if features >= 4 and 'intensity' in columns:
        result[:,3] = columns['intensity']
    if features >= 5:
        result[:,4] = columns.get('has_intensity', 1 if 'intensity' in columns else 0)
    return result
