import mmap
import os
import struct
import tempfile
import threading
import weakref


class Array:
    CHUNK_SIZE_BYTES = 4096

    # Header constants
    MAGIC = b"ARYF"
    HEADER_VERSION = 1
    HEADER_SIZE = 32
    HEADER_FORMAT = (
        "<4sHB8sIQ5x"  # magic(4), version(2), dtype_len(1), dtype(8), element_size(4), length(8), reserved(5)
    )

    def __init__(self, dtype, filename=None, mode="r+b", initial_elements=0):
        self._lock = threading.Lock()

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
        self._data_offset = self.HEADER_SIZE  # All data starts after header

        if "w" in mode or not os.path.exists(filename):
            # Create or truncate file
            self._file = open(filename, "w+b")
            self._len = 0
            self._allocate_capacity(initial_elements)
            self._write_header()
        else:
            # Open existing file
            self._file = open(filename, mode)
            if not self._read_header():
                raise ValueError("File does not have a valid array header")

            current_file_size = os.fstat(self._file.fileno()).st_size
            data_size = current_file_size - self.HEADER_SIZE

            # Calculate capacity based on current data size and ensure chunk alignment
            min_elements = (data_size + self._element_size - 1) // self._element_size
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

    def _validate_index(self, index):
        """Validate and normalize an index, returning the normalized value."""
        if not isinstance(index, int):
            raise TypeError("Index must be an integer")

        # Handle negative indices
        if index < 0:
            index = self._len + index

        if not (0 <= index < self._len):
            raise IndexError("Index out of bounds")

        return index

    def _pack_value(self, value):
        """Pack a value into bytes according to the dtype format."""
        try:
            return struct.pack(self._dtype_format, value)
        except struct.error as e:
            raise TypeError(f"Value {value} cannot be packed as {self._dtype_format}: {e}")

    def _write_header(self):
        """Write header to the beginning of the file."""
        dtype_bytes = self._dtype.encode("ascii")[:8]  # Limit to 8 bytes
        dtype_bytes = dtype_bytes.ljust(8, b"\x00")  # Pad with nulls

        header = struct.pack(
            self.HEADER_FORMAT,
            self.MAGIC,
            self.HEADER_VERSION,
            len(self._dtype),
            dtype_bytes,
            self._element_size,
            self._len,
        )

        self._file.seek(0)
        self._file.write(header)
        self._file.flush()

    def _read_header(self):
        """Read and validate header from file. Returns True if valid header, False if no header."""
        self._file.seek(0)
        header_data = self._file.read(self.HEADER_SIZE)

        if len(header_data) < self.HEADER_SIZE:
            return False

        magic, version, dtype_len, dtype_bytes, element_size, length = struct.unpack(self.HEADER_FORMAT, header_data)

        if magic != self.MAGIC:
            return False

        if version != self.HEADER_VERSION:
            raise ValueError(f"Unsupported header version: {version}")

        # Extract dtype string
        dtype = dtype_bytes[:dtype_len].decode("ascii")

        # Validate dtype matches
        if dtype != self._dtype:
            raise ValueError(f"File dtype '{dtype}' does not match requested dtype '{self._dtype}'")

        if element_size != self._element_size:
            raise ValueError(f"File element size {element_size} does not match expected {self._element_size}")

        self._len = length
        return True

    def __getitem__(self, index):
        index = self._validate_index(index)

        if not self._mmap:
            raise RuntimeError("Array is not memory-mapped. This should not happen if len > 0.")

        offset = self._data_offset + index * self._element_size
        data = self._mmap[offset : offset + self._element_size]
        return struct.unpack(self._dtype_format, data)[0]

    def __setitem__(self, index, value):
        index = self._validate_index(index)

        with self._lock:
            if not self._mmap:
                raise RuntimeError("Array is not memory-mapped. This should not happen if len > 0.")

            offset = self._data_offset + index * self._element_size
            packed_value = self._pack_value(value)
            self._mmap[offset : offset + self._element_size] = packed_value

    def append(self, value):
        with self._lock:
            if self._len == self._capacity:
                self._resize(self._len + 1)

            offset = self._data_offset + self._len * self._element_size
            packed_value = self._pack_value(value)

            self._mmap[offset : offset + self._element_size] = packed_value
            self._len += 1

    def _allocate_capacity(self, min_elements):
        """Allocate capacity for at least min_elements, rounded up to chunk boundary."""
        bytes_needed = min_elements * self._element_size + self.HEADER_SIZE
        chunks_needed = (bytes_needed + self.CHUNK_SIZE_BYTES - 1) // self.CHUNK_SIZE_BYTES
        total_file_size = chunks_needed * self.CHUNK_SIZE_BYTES
        self._capacity_bytes = total_file_size - self.HEADER_SIZE
        self._capacity = self._capacity_bytes // self._element_size
        self._file.truncate(total_file_size)

    def _resize(self, min_new_len):
        if self._mmap:
            self._mmap.close()

        self._allocate_capacity(min_new_len)
        self._mmap = mmap.mmap(self._file.fileno(), 0)

    def extend(self, iterable):
        values = list(iterable)
        num_new_elements = len(values)

        if num_new_elements == 0:
            return

        with self._lock:
            new_len = self._len + num_new_elements
            if new_len > self._capacity:
                self._resize(new_len)

            # Batch write all values directly to mmap
            offset = self._data_offset + self._len * self._element_size
            for value in values:
                packed_value = self._pack_value(value)
                self._mmap[offset : offset + self._element_size] = packed_value
                offset += self._element_size

            self._len = new_len

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

        with self._lock:
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
                original_len = self._len
                new_total_len = original_len * value

                # Resize if needed
                if new_total_len > self._capacity:
                    self._resize(new_total_len)

                # Copy data in-place
                src_offset = self._data_offset
                dst_offset = self._data_offset + original_len * self._element_size

                for _ in range(value - 1):
                    # Copy the original data chunk
                    chunk_size = original_len * self._element_size
                    self._mmap[dst_offset : dst_offset + chunk_size] = self._mmap[src_offset : src_offset + chunk_size]
                    dst_offset += chunk_size

                self._len = new_total_len
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
            # Update header with final length
            self._write_header()

            # Only truncate if the file was opened in a writable mode
            # and if the current size is greater than the actual data length
            current_file_size = os.fstat(self._file.fileno()).st_size
            actual_total_size = self.HEADER_SIZE + self._len * self._element_size
            if current_file_size > actual_total_size:
                self._file.truncate(actual_total_size)
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
