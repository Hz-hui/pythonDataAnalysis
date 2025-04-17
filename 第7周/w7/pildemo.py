from PIL import Image
import imagehash
import scipy
import sys


with Image.open("test.png") as im1:
	c_hist_1 = im1.histogram()
	print(len(c_hist_1))
	ahash_1 = imagehash.average_hash(im1)
	print(ahash_1)

with Image.open(sys.argv[1]) as im2:
	c_hist_2 = im2.histogram()
	ahash_2 = imagehash.average_hash(im2)

print(ahash_1-ahash_2)

p_cor = scipy.stats.pearsonr(c_hist_1, c_hist_2)
print(p_cor)

s_cor = scipy.stats.spearmanr(c_hist_1, c_hist_2)
print(s_cor)

k_cor = scipy.stats.kendalltau(c_hist_1, c_hist_2)
print(k_cor)

