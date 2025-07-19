import mmap
import os
import struct
import tempfile
import weakref


class Array:
    CHUNK_SIZE_BYTES = 4096

    def __init__(self, dtype, filename=None, mode="r+b", initial_elements=0):
        if filename is None:
            fd, filename = tempfile.mkstemp()
            os.close(fd)
            mode = "w+b"  # Always create new temp files

        self._filename = filename
        self._dtype = dtype
        self._dtype_format = dtype
        self._element_size = struct.calcsize(dtype)
        self._file = None
        self._mmap = None
        self._len = 0
        self._capacity = 0
        self._capacity_bytes = 0  # Initialize _capacity_bytes here

        if "w" in mode or not os.path.exists(filename):
            # Create or truncate file
            self._file = open(filename, "w+b")
            self._len = 0
            self._allocate_capacity(initial_elements)
        else:
            # Open existing file
            self._file = open(filename, mode)
            current_file_size = os.fstat(self._file.fileno()).st_size
            self._len = current_file_size // self._element_size  # Actual number of elements

            # Calculate capacity based on current file size and ensure chunk alignment
            min_elements = (current_file_size + self._element_size - 1) // self._element_size
            self._allocate_capacity(min_elements)

        # Only mmap if the file has a non-zero size
        if self._capacity_bytes > 0:
            self._mmap = mmap.mmap(self._file.fileno(), 0)
            
        # Set up finalizer to ensure cleanup even if close() isn't called
        self._finalizer = weakref.finalize(self, self.close)

    def __len__(self):
        return self._len

    def __iter__(self):
        current_len = self._len
        for i in range(current_len):
            yield self[i]

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("Index must be an integer")

        # Handle negative indices
        if index < 0:
            index = self._len + index

        if not (0 <= index < self._len):
            raise IndexError("Index out of bounds")
        if not self._mmap:
            raise RuntimeError("Array is not memory-mapped. This should not happen if len > 0.")

        offset = index * self._element_size
        data = self._mmap[offset : offset + self._element_size]
        return struct.unpack(self._dtype_format, data)[0]

    def __setitem__(self, index, value):
        if not isinstance(index, int):
            raise TypeError("Index must be an integer")

        # Handle negative indices
        if index < 0:
            index = self._len + index

        if not (0 <= index < self._len):
            raise IndexError("Index out of bounds")
        if not self._mmap:
            raise RuntimeError("Array is not memory-mapped. This should not happen if len > 0.")

        offset = index * self._element_size
        try:
            packed_value = struct.pack(self._dtype_format, value)
        except struct.error as e:
            raise TypeError(f"Value {value} cannot be packed as {self._dtype_format}: {e}")

        self._mmap[offset : offset + self._element_size] = packed_value

    def append(self, value):
        if self._len == self._capacity:
            self._resize(self._len + 1)

        offset = self._len * self._element_size
        try:
            packed_value = struct.pack(self._dtype_format, value)
        except struct.error as e:
            raise TypeError(f"Value {value} cannot be packed as {self._dtype_format}: {e}")

        if not self._mmap:
            # This happens when initial_elements=0 and we're appending first element
            self._allocate_capacity(1)
            self._mmap = mmap.mmap(self._file.fileno(), 0)

        self._mmap[offset : offset + self._element_size] = packed_value
        self._len += 1

    def _allocate_capacity(self, min_elements):
        """Allocate capacity for at least min_elements, rounded up to chunk boundary."""
        bytes_needed = min_elements * self._element_size
        chunks_needed = (bytes_needed + self.CHUNK_SIZE_BYTES - 1) // self.CHUNK_SIZE_BYTES
        self._capacity_bytes = chunks_needed * self.CHUNK_SIZE_BYTES
        self._capacity = self._capacity_bytes // self._element_size
        self._file.truncate(self._capacity_bytes)

    def _resize(self, min_new_len):
        if self._mmap:
            self._mmap.close()

        self._allocate_capacity(min_new_len)
        self._mmap = mmap.mmap(self._file.fileno(), 0)

    def extend(self, iterable):
        values = list(iterable)
        num_new_elements = len(values)

        if self._len + num_new_elements > self._capacity:
            self._resize(self._len + num_new_elements)

        for value in values:
            self.append(value)

    def __contains__(self, value):
        for i in range(self._len):
            if self[i] == value:
                return True
        return False

    def __iadd__(self, other):
        if hasattr(other, "__iter__"):
            self.extend(other)
            return self
        return NotImplemented

    def __imul__(self, value):
        if not isinstance(value, int) or value < 0:
            return NotImplemented

        if value == 0:
            self._len = 0
            if self._mmap:
                self._mmap.close()
                self._mmap = None
            if self._file:
                self._file.truncate(0)
            self._capacity = 0
            self._capacity_bytes = 0
        elif value > 1:
            original_elements = [self[i] for i in range(len(self))]
            for _ in range(value - 1):
                self.extend(original_elements)
        return self

    def flush(self):
        if self._mmap:
            self._mmap.flush()

    def close(self):
        if self._mmap:
            # Ensure all writes are on disk before truncating
            self._mmap.flush()
            self._mmap.close()
            self._mmap = None

        if self._file:
            # Only truncate if the file was opened in a writable mode
            # and if the current size is greater than the actual data length
            current_file_size = os.fstat(self._file.fileno()).st_size
            actual_data_bytes = self._len * self._element_size
            if current_file_size > actual_data_bytes:
                self._file.truncate(actual_data_bytes)
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
