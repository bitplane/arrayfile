import gc
import io
import os
import struct
import tempfile
import weakref

import pytest

from arrayfile import Array


@pytest.fixture
def temp_filepath():
    fd, path = tempfile.mkstemp()
    os.close(fd)
    yield path
    os.remove(path)


def test_init_new_file_no_filename():
    with Array("i") as array:
        filename = array._filename
        assert os.path.exists(filename)
        assert array._dtype == "i"
        assert array._element_size == struct.calcsize("i")
        assert len(array) == 0
    assert not os.path.exists(filename)


@pytest.mark.parametrize(
    "dtype, initial_elements, expected_element_size",
    [
        ("b", 0, 1),
        ("i", 0, 4),
        ("d", 0, 8),
        ("i", 10, 4),
        ("f", 5, 4),
    ],
)
def test_init_new_file_with_filename(temp_filepath, dtype, initial_elements, expected_element_size):
    array = Array(dtype, temp_filepath, "w+b", initial_elements)
    assert array._filename == temp_filepath
    assert array._dtype == dtype
    assert array._element_size == expected_element_size
    assert len(array) == 0
    array.close()


def test_init_existing_file(temp_filepath):
    # Create a file with array data using proper header format
    with Array("i", temp_filepath, "w+b") as array:
        array.append(10)
        array.append(20)
        array.append(30)

    # Reopen and verify
    array = Array("i", temp_filepath, "r+b")
    assert len(array) == 3
    assert array[0] == 10
    assert array[1] == 20
    assert array[2] == 30
    array.close()


def test_init_existing_file_read_only_does_not_modify_file(temp_filepath):
    with Array("i", temp_filepath, "w+b") as array:
        array.extend([10, 20, 30])

    original_size = os.path.getsize(temp_filepath)
    with Array("i", temp_filepath, "rb") as array:
        assert list(array) == [10, 20, 30]
        with pytest.raises(io.UnsupportedOperation, match="not writable"):
            array.append(40)

    assert os.path.getsize(temp_filepath) == original_size


def test_append_and_len(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    assert len(array) == 0

    array.append(1)
    assert len(array) == 1
    assert array[0] == 1

    array.append(2)
    assert len(array) == 2
    assert array[1] == 2
    array.close()


def test_append_triggers_resize(temp_filepath):
    # Test that we can append many elements and the array grows
    array = Array("B", temp_filepath, "w+b", 0)

    # Append enough elements to trigger multiple resizes
    num_elements = 5000  # Should be enough to trigger resize
    for i in range(num_elements):
        array.append(i % 256)  # Ensure value is within 'B' range
        assert len(array) == i + 1

    # Verify all values are correct
    for i in range(num_elements):
        assert array[i] == i % 256

    array.close()


def test_append_type_error(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    with pytest.raises(TypeError, match="cannot be packed"):
        array.append("not an int")
    array.close()


def test_append_empty_file_triggers_mmap_creation(temp_filepath):
    # Test that we can append to an empty array
    array = Array("i", temp_filepath, "w+b", initial_elements=0)

    # Append to empty array
    array.append(42)

    # Verify the data
    assert len(array) == 1
    assert array[0] == 42

    array.close()


def test_getitem_valid(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(100)
    array.append(200)
    assert array[0] == 100
    assert array[1] == 200
    array.close()


def test_getitem_out_of_bounds(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(10)
    with pytest.raises(IndexError, match="Index out of bounds"):
        _ = array[1]
    with pytest.raises(IndexError, match="Index out of bounds"):
        _ = array[-2]  # -2 is out of bounds for 1-element array
    array.close()


def test_getitem_type_error(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(10)
    with pytest.raises(TypeError, match="Index must be an integer"):
        _ = array[0.5]
    array.close()


def test_getitem_no_mmap_after_close(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(10)
    array.close()
    with pytest.raises(RuntimeError, match="Array is not memory-mapped"):
        _ = array[0]


def test_setitem_valid(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(100)
    array.append(200)
    array[0] = 150
    assert array[0] == 150
    array[1] = 250
    assert array[1] == 250
    array.close()


def test_setitem_out_of_bounds(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(10)
    with pytest.raises(IndexError, match="Index out of bounds"):
        array[1] = 20
    with pytest.raises(IndexError, match="Index out of bounds"):
        array[-2] = 5  # -2 is out of bounds for 1-element array
    array.close()


def test_setitem_type_error(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(10)
    with pytest.raises(TypeError, match="Index must be an integer"):
        array[0.5] = 20
    with pytest.raises(TypeError, match="cannot be packed"):
        array[0] = "not an int"
    array.close()


def test_setitem_no_mmap_after_close(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(10)
    array.close()
    with pytest.raises(RuntimeError, match="Array is not memory-mapped"):
        array[0] = 20


def test_flush(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(1)
    array.append(2)
    array.flush()
    array.close()  # Close the array to trigger truncation

    # Data should be on disk, and file size should be truncated
    array_reopen = Array("i", temp_filepath, "r+b")
    assert len(array_reopen) == 2
    assert array_reopen[0] == 1
    assert array_reopen[1] == 2
    array_reopen.close()


def test_flush_persists_length_in_header(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(42)
    array.flush()

    with open(temp_filepath, "rb") as file:
        header = file.read(Array.HEADER_SIZE)

    *_, length = struct.unpack(Array.HEADER_FORMAT, header)
    assert length == 1
    array.close()


def test_close_truncates(temp_filepath):
    # Test that data persists after close
    array = Array("i", temp_filepath, "w+b", 100)
    array.append(1)
    array.append(2)
    array.close()

    # Reopen and verify data persists
    array2 = Array("i", temp_filepath, "r+b")
    assert len(array2) == 2
    assert array2[0] == 1
    assert array2[1] == 2
    array2.close()


def test_close_multiple_times(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(1)
    array.close()
    array.close()  # Should not raise error


def test_finalizer_releases_array_and_persists_data(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(42)
    reference = weakref.ref(array)

    del array
    gc.collect()

    assert reference() is None
    with Array("i", temp_filepath, "rb") as reopened:
        assert list(reopened) == [42]


def test_context_manager(temp_filepath):
    with Array("i", temp_filepath, "w+b", 100) as array:
        array.append(10)
        array.append(20)

    # After exiting context, verify data persists
    assert os.path.exists(temp_filepath)

    # Verify content by reopening
    array_reopen = Array("i", temp_filepath, "r+b")
    assert len(array_reopen) == 2
    assert array_reopen[0] == 10
    assert array_reopen[1] == 20
    array_reopen.close()


def test_persistence(temp_filepath):
    with Array("i", temp_filepath, "w+b") as array:
        for i in range(100):
            array.append(i)

    with Array("i", temp_filepath, "r+b") as array_reopen:
        assert len(array_reopen) == 100
        for i in range(100):
            assert array_reopen[i] == i


@pytest.mark.parametrize(
    "dtype, test_value",
    [
        ("b", 127),
        ("B", 255),
        ("h", 32767),
        ("H", 65535),
        ("i", 2147483647),
        ("I", 4294967295),
        ("l", 2147483647),
        ("L", 4294967295),
        ("q", 9223372036854775807),
        ("Q", 18446744073709551615),
        ("f", 3.14159),
        ("d", 2.718281828459045),
    ],
)
def test_different_dtypes(temp_filepath, dtype, test_value):
    array = Array(dtype, temp_filepath, "w+b")
    array.append(test_value)
    assert len(array) == 1
    expected = pytest.approx(test_value) if dtype in ["f", "d"] else test_value
    assert array[0] == expected
    array.close()


def test_empty_array_access(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    assert len(array) == 0
    with pytest.raises(IndexError):
        _ = array[0]
    with pytest.raises(IndexError):
        array[0] = 1
    array.close()


def test_contains(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(10)
    array.append(20)
    array.append(30)
    assert 10 in array
    assert 20 in array
    assert 30 in array
    assert 40 not in array
    assert 5 not in array
    array.close()

    array.close()


def test_iteration_non_empty(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    elements = [10, 20, 30]
    array.extend(elements)
    assert list(array) == elements
    array.close()


def test_iteration_empty(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    assert list(array) == []
    array.close()


def test_extend(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.extend([1, 2, 3])
    assert len(array) == 3
    assert array[0] == 1
    assert array[1] == 2
    assert array[2] == 3

    array.extend([4, 5])
    assert len(array) == 5
    assert array[3] == 4
    assert array[4] == 5
    array.close()


def test_iadd(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(1)
    array += [2, 3, 4]
    assert len(array) == 4
    assert array[0] == 1
    assert array[1] == 2
    assert array[2] == 3
    assert array[3] == 4
    array.close()


def test_imul(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.extend([1, 2])
    array *= 3
    assert len(array) == 6
    assert array[0] == 1
    assert array[1] == 2
    assert array[2] == 1
    assert array[3] == 2
    assert array[4] == 1
    assert array[5] == 2
    array.close()


def test_imul_not_implemented(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.append(1)
    # Test with non-integer value
    result = array.__imul__(1.5)
    assert result is NotImplemented
    # Test with negative value
    result = array.__imul__(-1)
    assert result is NotImplemented
    array.close()


def test_imul_zero(temp_filepath):
    array = Array("i", temp_filepath, "w+b")
    array.extend([1, 2, 3])
    array *= 0
    assert len(array) == 0
    # Verify file is truncated to 0 bytes
    assert os.path.getsize(temp_filepath) == 0
    array.close()


def test_iadd_not_implemented():
    """Test __iadd__ with non-iterable - line 142."""
    array = Array("i", None, "w+b")  # Use temp file

    try:
        # Test with non-iterable value
        result = array.__iadd__(42)  # Number is not iterable
        assert result is NotImplemented

    finally:
        array.close()


def test_negative_index_access(temp_filepath):
    """Test that negative indices work correctly in Array."""
    array = Array("i", temp_filepath, "w+b")
    array.append(10)
    array.append(20)
    array.append(30)

    # Test positive indices work
    assert array[0] == 10
    assert array[1] == 20
    assert array[2] == 30

    # Test negative indices (this should work but currently fails)
    assert array[-1] == 30  # Should be last element
    assert array[-2] == 20  # Should be second-to-last element
    assert array[-3] == 10  # Should be first element

    array.close()


def test_negative_index_out_of_bounds(temp_filepath):
    """Test that negative indices properly check bounds."""
    array = Array("i", temp_filepath, "w+b")
    array.append(10)
    array.append(20)

    # These should raise IndexError
    with pytest.raises(IndexError, match="Index out of bounds"):
        _ = array[-3]  # Only 2 elements, so -3 is out of bounds

    with pytest.raises(IndexError, match="Index out of bounds"):
        _ = array[-10]  # Way out of bounds

    array.close()


def test_array_length_after_reopen_with_preallocation(temp_filepath):
    """Test that Array length is correct after reopening a file with pre-allocated space."""
    # Create array with initial capacity but only add a few elements
    array = Array("i", temp_filepath, "w+b", initial_elements=1000)  # Pre-allocate space for 1000 elements
    array.append(10)
    array.append(20)
    array.append(30)

    # Verify length is correct
    assert len(array) == 3
    assert array[0] == 10
    assert array[1] == 20
    assert array[2] == 30
    assert array[-1] == 30  # Last element should be 30

    array.close()

    # Reopen the array - it should remember the correct length
    array2 = Array("i", temp_filepath, "r+b")
    assert len(array2) == 3
    assert array2[0] == 10
    assert array2[1] == 20
    assert array2[2] == 30
    assert array2[-1] == 30  # Last element should still be 30, not 0

    array2.close()


def test_invalid_header_file(temp_filepath):
    """Test opening a file without valid array header."""
    # Create a file with non-array data
    with open(temp_filepath, "wb") as f:
        f.write(b"This is not an array file")

    with pytest.raises(ValueError, match="File does not have a valid array header"):
        Array("i", temp_filepath, "r+b")


def test_truncated_header_file(temp_filepath):
    """Test opening a file with truncated header."""
    # Create a file with only partial header
    with open(temp_filepath, "wb") as f:
        f.write(b"ARYF")  # Only magic, missing rest of header

    with pytest.raises(ValueError, match="File does not have a valid array header"):
        Array("i", temp_filepath, "r+b")


def test_corrupted_header_file(temp_filepath):
    """Test opening a file with corrupted header that causes unsupported version."""
    # Create a file with 32 bytes that will parse but give invalid version
    with open(temp_filepath, "wb") as f:
        f.write(b"ARYF" + b"\xff" * 28)  # Magic + garbage giving version 65535

    with pytest.raises(ValueError, match="Unsupported header version: 65535"):
        Array("i", temp_filepath, "r+b")


def test_wrong_magic_number(temp_filepath):
    """Test opening a file with wrong magic number."""
    # Create a file with wrong magic but correct size
    with open(temp_filepath, "wb") as f:
        f.write(b"WRNG" + b"\x00" * 28)  # Wrong magic + padding

    with pytest.raises(ValueError, match="File does not have a valid array header"):
        Array("i", temp_filepath, "r+b")


def test_unsupported_version(temp_filepath):
    """Test opening a file with unsupported version."""
    # Create an array file then manually modify the version
    with Array("i", temp_filepath, "w+b") as arr:
        arr.append(42)

    # Modify the version in the header
    with open(temp_filepath, "r+b") as f:
        f.seek(4)  # Position after magic
        f.write(struct.pack("<H", 999))  # Write unsupported version

    with pytest.raises(ValueError, match="Unsupported header version: 999"):
        Array("i", temp_filepath, "r+b")


def test_dtype_mismatch(temp_filepath):
    """Test opening a file with different dtype than expected."""
    # Create an array with 'i' dtype
    with Array("i", temp_filepath, "w+b") as arr:
        arr.append(42)

    # Try to open with 'd' dtype
    with pytest.raises(ValueError, match="File dtype 'i' does not match requested dtype 'd'"):
        Array("d", temp_filepath, "r+b")


def test_dtype_too_long_for_header_is_rejected(temp_filepath):
    dtype = "        i"
    assert struct.calcsize(dtype) == struct.calcsize("i")

    with pytest.raises(ValueError, match="dtype must be at most 8 ASCII bytes"):
        Array(dtype, temp_filepath, "w+b")


def test_element_size_mismatch(temp_filepath):
    """Test opening a file with mismatched element size."""
    # Create an array file then manually modify the element size
    with Array("i", temp_filepath, "w+b") as arr:
        arr.append(42)

    # Modify the element size in the header
    with open(temp_filepath, "r+b") as f:
        f.seek(4 + 2 + 1 + 8)  # Position after magic + version + dtype_len + dtype
        f.write(struct.pack("<I", 999))  # Write wrong element size

    with pytest.raises(ValueError, match="File element size 999 does not match expected 4"):
        Array("i", temp_filepath, "r+b")


def test_length_exceeding_file_data_is_rejected(temp_filepath):
    with Array("i", temp_filepath, "w+b") as array:
        array.append(42)

    with open(temp_filepath, "r+b") as file:
        length_offset = struct.calcsize("<4sHB8sI")
        file.seek(length_offset)
        file.write(struct.pack("<Q", 1000))

    with pytest.raises(ValueError, match="declares 1000 elements, but file only contains 1"):
        Array("i", temp_filepath, "r+b")


def test_extend_empty_list(temp_filepath):
    """Test extending with empty list."""
    array = Array("i", temp_filepath, "w+b")
    array.append(1)
    array.extend([])  # This should trigger line 205
    assert len(array) == 1
    assert array[0] == 1
    array.close()


def test_extend_triggers_resize(temp_filepath):
    """Test that extend can trigger resize."""
    array = Array("i", temp_filepath, "w+b")
    # Add many elements to trigger resize during extend
    large_list = list(range(2000))
    array.extend(large_list)  # This should trigger line 210
    assert len(array) == 2000
    for i in range(2000):
        assert array[i] == i
    array.close()


def test_imul_triggers_resize(temp_filepath):
    """Test that __imul__ can trigger resize."""
    array = Array("i", temp_filepath, "w+b")
    # Create a large array and multiply to trigger resize
    array.extend(list(range(1000)))
    array *= 3  # This should trigger line 253
    assert len(array) == 3000
    # Verify pattern repeats correctly
    for i in range(1000):
        assert array[i] == i
        assert array[i + 1000] == i
        assert array[i + 2000] == i
    array.close()


@pytest.mark.parametrize("mode", ["ab", "a+b", "ab+"])
def test_append_mode_rejected_without_modifying_file(tmp_path, mode):
    path = tmp_path / "array"
    with Array("i", path, "w+b") as array:
        array.append(1)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Append mode"):
        Array("i", path, mode)
    assert path.read_bytes() == original


@pytest.mark.parametrize("failure", ["allocation", "mapping"])
def test_resize_failure_preserves_existing_values(tmp_path, monkeypatch, failure):
    import arrayfile.array as implementation

    path = tmp_path / "array"
    array = Array("i", path, "w+b")
    array.append(42)
    file = array._file
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise OSError("allocation failed")

        if failure == "allocation":
            patch.setattr(array, "_allocate_capacity", fail)
        else:
            patch.setattr(implementation.mmap, "mmap", fail)
        with pytest.raises(OSError, match="allocation failed"):
            array.extend(range(2000))
    array.close()
    assert file.closed
    with Array("i", path, "rb") as reopened:
        assert list(reopened) == [42]


def test_close_releases_handles_after_flush_error(tmp_path):
    from unittest.mock import Mock

    array = Array("i", tmp_path / "array", "w+b")
    file = array._file
    array._mmap.close()
    mapping = Mock()
    mapping.flush.side_effect = OSError("flush failed")
    array._finalizer_state["mmap"] = mapping
    with pytest.raises(OSError, match="flush failed"):
        array.close()
    mapping.close.assert_called_once()
    assert file.closed
    array.close()


@pytest.mark.parametrize("mode", ["rb", "r+b", "rb+"])
def test_read_modes_require_existing_file(tmp_path, mode):
    path = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        Array("i", path, mode)
    assert not path.exists()


def test_exclusive_create_preserves_existing_file(tmp_path):
    path = tmp_path / "array"
    with Array("i", path, "xb") as array:
        array.append(42)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        Array("i", path, "x+b")
    assert path.read_bytes() == original


@pytest.mark.parametrize("mode", ["r", "invalid", "wwb", None])
def test_invalid_mode_does_not_create_file(tmp_path, mode):
    path = tmp_path / "array"
    with pytest.raises(ValueError, match="mode"):
        Array("i", path, mode)
    assert not path.exists()


@pytest.mark.parametrize("empty_state", ["new", "reopened", "cleared"])
def test_multiply_empty_array_then_append(tmp_path, empty_state):
    path = tmp_path / "array"
    array = Array("i", path, "w+b")
    if empty_state == "reopened":
        array.close()
        array = Array("i", path)
    elif empty_state == "cleared":
        array.append(1)
        array *= 0
    with array:
        array *= 2
        assert list(array) == []
        array.append(42)
    with Array("i", path, "rb") as reopened:
        assert list(reopened) == [42]


@pytest.mark.parametrize("dtype", ["", "0i", "x", "2i", "ii", "0s", "x0p", "invalid"])
def test_invalid_dtype_preserves_destination(tmp_path, dtype):
    path = tmp_path / "array"
    path.write_bytes(b"existing contents")
    with pytest.raises(ValueError, match="dtype"):
        Array(dtype, path, "w+b")
    assert path.read_bytes() == b"existing contents"


@pytest.mark.parametrize("dtype,value", [("<i", 42), ("xi", 42), ("3s", b"abc"), ("4p", b"abc")])
def test_single_value_struct_formats_round_trip(tmp_path, dtype, value):
    path = tmp_path / "array"
    with Array(dtype, path, "w+b") as array:
        array.append(value)
    with Array(dtype, path, "rb") as array:
        assert array[0] == value


@pytest.mark.parametrize("cleanup", ["close", "gc"])
def test_temporary_array_removes_file(cleanup):
    array = Array("i")
    array.append(42)
    filename = array._filename
    reference = weakref.ref(array)
    if cleanup == "close":
        array.close()
        array.close()
    else:
        del array
        gc.collect()
        assert reference() is None
    assert not os.path.exists(filename)


def test_failed_temporary_initialization_removes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def fail(*args):
        raise OSError("allocation failed")

    monkeypatch.setattr(Array, "_allocate_capacity", fail)
    with pytest.raises(OSError, match="allocation failed"):
        Array("i")
    assert list(tmp_path.iterdir()) == []
