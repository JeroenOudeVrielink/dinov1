c = 2048
size = 7
reduction_factor = 8

print(f"Start: c={c}, size={size}x{size} total={c*size*size}")

c1 = c // (reduction_factor**1)
size = size * 2
print(f"Step 1: c={c1}, size={size}x{size} total={c1*size*size}")

c2 = c // (reduction_factor**2)
size = size * 2
print(f"Step 2: c={c2}, size={size}x{size} total={c2*size*size}")

c3 = c // (reduction_factor**3)
size = size * 2
print(f"Step 3: c={c3}, size={size}x{size} total={c3*size*size}")

c4 = 1
size = size * 4
print(f"Final: c={c4}, size={size}x{size} total={c4*size*size}")
