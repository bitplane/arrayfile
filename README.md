# arrayfile

A file-backed numeric array using struct.pack. Does not support inserts or
slicing.

Smaller than relying on numpy though.

## Installation

```bash
pip install arrayfile
```

## Usage

### Temporary Array

```python
from arrayfile import Array

# Create a temporary array with float data
with Array('f') as arr:
    arr.append(3.14)
    arr.append(2.71)
    arr.extend([1.41, 1.73])

    print(f"Length: {len(arr)}")
    print(f"Values: {[arr[i] for i in range(len(arr))]}")
```

### Persistent Array

```python
from arrayfile import Array

# Create and populate an array file
with Array('i', 'numbers.array', 'w+b') as arr:
    for i in range(1000):
        arr.append(i * 2)

# Reopen the same file later
with Array('i', 'numbers.array', 'r+b') as arr:
    print(f"Array has {len(arr)} elements")
    print(f"First element: {arr[0]}")
    print(f"Last element: {arr[-1]}")

    # Add more data
    arr.append(2000)
```
