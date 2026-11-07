"""
Empirical Verification of Marlin INT4 x FP16 Bit Manipulation Mechanics
Focus:
1. IEEE 754 half-precision exponent injection: 0x6400 (1024.0).
2. Dequantization correctness for all 16 integer nibbles [0, 15].
3. Analysis of bias subtraction (1024.0 vs 1032.0 for signed/zero-point INT4).
4. Register packing & extraction of all 8 nibbles: shifting requirements and ALU instruction overhead.
"""

import struct
import numpy as np

def fp16_from_bits(u16: int) -> float:
    """Interpret 16-bit integer as IEEE 754 half-precision float."""
    raw_bytes = struct.pack('<H', u16)
    return np.frombuffer(raw_bytes, dtype=np.float16)[0]

def bits_from_fp16(val: float) -> int:
    """Get 16-bit unsigned integer representation of FP16 float."""
    f16 = np.float16(val)
    raw_bytes = f16.tobytes()
    return struct.unpack('<H', raw_bytes)[0]

def test_single_nibble_unpack():
    print("=" * 60)
    print("TEST 1: Marlin Single Nibble Exponent Injection (0x6400)")
    print("=" * 60)
    
    magic_constant = 0x6400  # IEEE 754 FP16 representation of 1024.0
    magic_float = fp16_from_bits(magic_constant)
    print(f"Magic Constant: 0x{magic_constant:04X} -> Float Value: {magic_float}")
    assert magic_float == 1024.0, f"Expected 1024.0, got {magic_float}"

    # Verify IEEE 754 fields of 0x6400:
    # sign = bit 15 (0)
    # exponent = bits 14-10 (11001_2 = 25_10)
    # mantissa = bits 9-0 (0000000000_2 = 0)
    # Value = 2^(25 - 15) * (1 + 0/1024) = 2^10 = 1024.0
    sign = (magic_constant >> 15) & 1
    exponent = (magic_constant >> 10) & 0x1F
    mantissa = magic_constant & 0x3FF
    print(f"0x6400 breakdown: sign={sign}, biased_exp={exponent} (unbiased={exponent - 15}), mantissa={mantissa}")

    all_exact = True
    print("\nTesting all 16 INT4 values (unsigned 0 to 15):")
    for v in range(16):
        # Place nibble v into low 4 bits [3:0]
        injected_bits = magic_constant | v
        injected_float = fp16_from_bits(injected_bits)
        
        # Subtraction step: injected_float - 1024.0 in FP16 arithmetic
        decompressed_float = np.float16(injected_float) - np.float16(1024.0)
        
        expected = float(v)
        is_exact = (float(decompressed_float) == expected)
        if not is_exact:
            all_exact = False
        print(f"  v={v:2d} (0x{v:X}) -> Injected: 0x{injected_bits:04X} ({injected_float:7.1f}) "
              f"- 1024.0 = {decompressed_float:4.1f} | Exact? {is_exact}")

    assert all_exact, "Test failed: not all unsigned nibbles unpacked exactly!"
    print("\nResult: All 16 unsigned INT4 nibbles unpack EXACTLY without precision loss.")

def test_signed_int4_and_zeropoint():
    print("\n" + "=" * 60)
    print("TEST 2: Signed INT4 & Zero-Point Handling")
    print("=" * 60)
    # In LLM quantization (e.g. GPTQ / AWQ / Marlin), INT4 is either:
    # A) Signed two's complement: -8 to +7
    # B) Unsigned with zero_point = 8: raw stored value is (v + 8) in [0, 15]
    
    # If weights are stored with zero-point 8:
    # If we subtract 1024.0: (1024 + (v + 8)) - 1024 = v + 8 (NOT zero-centered!)
    # To get zero-centered v in [-8, +7], what must we subtract?
    # (1024 + (v + 8)) - (1024 + 8) = (1024 + v_stored) - 1032.0
    magic_constant = 0x6400
    subtrahend_zeropoint = 1032.0  # 1024.0 + 8.0
    subtrahend_bits = bits_from_fp16(subtrahend_zeropoint)
    print(f"Zero-point subtrahend: {subtrahend_zeropoint} -> FP16 bits: 0x{subtrahend_bits:04X}")

    print("\nTesting signed dequantization with zero-point subtraction (1032.0):")
    for signed_v in range(-8, 8):
        stored_nibble = signed_v + 8  # in [0, 15]
        injected_bits = magic_constant | stored_nibble
        injected_float = fp16_from_bits(injected_bits)
        decompressed = np.float16(injected_float) - np.float16(subtrahend_zeropoint)
        assert float(decompressed) == float(signed_v), f"Failed for {signed_v}"
        print(f"  true_v={signed_v:2d}, stored={stored_nibble:2d} -> "
              f"({injected_float:7.1f}) - {subtrahend_zeropoint} = {decompressed:4.1f} | Match: True")

    print("\nKey Finding: Document Section 5.3.3 states decompressed_val = sub.f16x2(reg, 1024.0).")
    print("That only works for UNSIGNED [0, 15] or symmetric without zero-point!")
    print("For standard zero-point = 8 quantization, the subtrahend MUST be 1032.0 (0x64106410)!")

def test_full_register_32bit_unpack_hazards():
    print("\n" + "=" * 60)
    print("TEST 3: Register Packing & 32-bit Unpack Hazards")
    print("=" * 60)
    # A 32-bit register holds 8 nibbles:
    # Bits [3:0]: Nibble 0
    # Bits [7:4]: Nibble 1
    # Bits [11:8]: Nibble 2
    # Bits [15:12]: Nibble 3
    # Bits [19:16]: Nibble 4
    # Bits [23:20]: Nibble 5
    # Bits [27:24]: Nibble 6
    # Bits [31:28]: Nibble 7

    # When treated as two 16-bit FP16 values:
    # Low FP16: bits [15:0]. Mantissa is bits [9:0].
    # High FP16: bits [31:16]. Mantissa is bits [25:16].
    
    # Notice:
    # In Low FP16: Nibble 0 is at bits [3:0] (correct position for 1.0 weight)
    # In High FP16: Nibble 4 is at bits [19:16] (which is bits [3:0] of High FP16!)
    
    # What about Nibble 1 (bits [7:4])?
    # If we mask Nibble 1 with 0x00F0 and OR with 0x6400:
    nibble_1 = 5
    packed = (nibble_1 << 4)
    masked_no_shift = (packed & 0x00F0) | 0x6400
    val_no_shift = fp16_from_bits(masked_no_shift)
    sub_no_shift = np.float16(val_no_shift) - np.float16(1024.0)
    print(f"Nibble 1 (value={nibble_1}) masked without shift: FP16 bits: 0x{masked_no_shift:04X}")
    print(f"  Value: {val_no_shift} -> After sub 1024.0: {sub_no_shift}")
    print(f"  Expected: {nibble_1}, Actual: {sub_no_shift} (Error: scaled by 16x = {nibble_1 * 16}!)")

    # To get correct value, Nibble 1 MUST be shifted right by 4 bits!
    shifted = ((packed >> 4) & 0x000F) | 0x6400
    val_shifted = fp16_from_bits(shifted)
    sub_shifted = np.float16(val_shifted) - np.float16(1024.0)
    print(f"Nibble 1 shifted right by 4: FP16 bits: 0x{shifted:04X} -> After sub 1024.0: {sub_shifted}")
    assert sub_shifted == nibble_1

    print("\nCRITICAL AUDIT FINDING on Section 5.3.3:")
    print("The document claims: 'In just two instructions (lop3 and sub.f16x2), Marlin completely unpacks")
    print("two 4-bit values into native IEEE FP16 floating-point numbers without executing a single scalar shift...'")
    print("While lop3 CAN extract (Nibble 0, Nibble 4) without a shift, extracting Nibbles (1, 5), (2, 6), (3, 7)")
    print("CANNOT be done with lop3 alone because lop3 has NO bit-shift capability!")
    print("They require either PTX 'shr.b32' (shifts) or byte permute 'prmt.b32' instructions!")

if __name__ == '__main__':
    test_single_nibble_unpack()
    test_signed_int4_and_zeropoint()
    test_full_register_32bit_unpack_hazards()
