import numpy as np
# Utility generator for all possible configurations
def gen_confs(item_types, count, idx, conf, size):
  if idx >= len(count):
    if conf.any():
      yield conf
    return
  c = count[idx];
  for a in range(c + 1):
    conf[idx] = a
    g = gen_confs(item_types, count, idx + 1, conf, size)
    yield from g
    size += item_types[idx]
    if size > 1:
      break

# Dynamic programming solver
def bin_pack_optimal(item_types, count, cache):
  key = tuple(count.tolist())
  if key in cache:
    return cache[key]
  n = len(item_types)
  total = count.sum()
  if not np.any(count):
    return np.zeros((0, n), dtype=np.int64)
  min_l = total + 1
  for conf in gen_confs(item_types, count, 0, np.zeros(n, dtype=np.int64), 0):
    pack = bin_pack_optimal(item_types, count - conf, cache)
    l = len(pack)
    if l < min_l:
      min_l = l
      min_pack = pack
      min_conf = conf.copy()
  ret = np.concatenate((min_pack, min_conf[None, :]))
  cache[key] = ret
  return ret

# Main bin packing function
# Assumes items are sorted in ascending order
def bin_pack(items, eps):
  n = len(items)
  size = items.sum()
  gamma = eps / 2
  large_idx = np.searchsorted(items, gamma, side='right')
  large_num = n - large_idx
  k = min(int(eps * size), large_num)
  if k == 0: # Just solve optimally in this case
    return bin_pack_optimal(items, np.ones(n, dtype=np.int64), {})
  rem = large_num % k
  items_ = items[-k-1:large_idx-1:-k][::-1].copy()
  items_count = np.full_like(items_, k, dtype=np.int64)
  if rem > 0:
    items_count[0] = rem
  # Optimally solve k grouped large items
  optimal_bin_pack = bin_pack_optimal(items_, items_count, {})
  l_ = len(optimal_bin_pack)
  l = l_ + k
  # Use optimal packing to pack the large elements
  bin_pack = np.zeros((l, n), dtype=np.int64)
  # Put largest k elements in own bin
  bin_pack[:k, -k:] = np.eye(k)
  # Put in rest of large elements
  for i in range(len(items_)):
    s = 0
    for j, b in enumerate(optimal_bin_pack[:, -i-1]):
      for g in range(s, s + b):
        bin_pack[j + k, -((i + 1) * k + g + 1)] = 1
      s += b
  bin_pack_sizes = (bin_pack * items).sum(axis=1)
  # First fit for small items
  for i, s in enumerate(items[large_idx-1::-1]):
    for j, b in enumerate(bin_pack):
      if s + bin_pack_sizes[j] <= 1:
        b[large_idx - 1 - i] = 1
        bin_pack_sizes[j] += s
        break
    else:
      bin_pack = np.concatenate((bin_pack, np.eye(1, n, large_idx - 1 - i, dtype=np.int64)))
      bin_pack_sizes = np.append(bin_pack_sizes, s)
  return bin_pack