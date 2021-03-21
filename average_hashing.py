#!/usr/bin/env python

# import standard modules
import math

# import third party modules
import numpy as np
from PIL import Image
from imagehash import ImageHash
from scipy.signal import convolve2d

# import project related modules
from matrix import MatrixUtil


class ImageHashing:

	#  From Wikipedia: standard RGB to luminance (the 'Y' in 'YUV').
	#  scalar values to convert pixel values to greyscale

	LUMA_FROM_R_COEFF = float(0.299)
	LUMA_FROM_G_COEFF = float(0.587)
	LUMA_FROM_B_COEFF = float(0.114)

	#  Since FINd uses 64x64 blocks, 1/64th of the image height/width
	#  respectively is a full block.
	FIND_WINDOW_SIZE_DIVISOR = 64

	def __init__(self):
		# TODO: write proper comment and class description
		"""
		See also comments on dct64To16. Input is (0..63)x(0..63);
		output is (1..16)x(1..16) with the latter indexed as (0..15)x(0..15).
		Returns 16x64 matrix.
		"""
		self.dct_matrix = self.compute_dct_matrix()

	@staticmethod
	def compute_dct_matrix() -> list:
		"""
		Function to create an initial Matrix. This Matrix is basically just a nested list element. Could be
		converted to a numpy array

		return: nested list element
		"""

		# create list with 16 elements
		d = [0] * 16

		for i in range(0, 16):

			# create list with 64 elements
			di = [0] * 64

			for j in range(0, 64):

				# insert math.cos for each element of the list
				di[j] = math.cos((math.pi / 2 / 64.0) * (i + 1) * (2 * j + 1))

			# create list of lists
			d[i] = di

		return d

	def from_file(self, filepath: str) -> Image:
		"""
		opens and image, resize it and returns the output from find_hash_256_from_float_luma method

		:param filepath: string path of the image

		:return: numpy array with image pixel values in each channel
		"""

		img = None
		try:
			img = Image.open(filepath)

		except IOError as e:
			raise e 

		return self.from_image(img)

	def from_image(self, img: Image) -> ImageHash:
		"""
		takes copy of loaded image and resizes it
		"""
		try:
			# create a copy of the image array to ensure a different memory pointer
			img = img.copy()

			# https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail
			# harmonize the image size, this is important for the further processing since we assume the same image
			# size vor the window operations
			img.thumbnail((512, 512))

		except IOError as e:
			raise e

		# conversion to numpy and extraction of matrix dimensions
		num_cols, num_rows = np.take(img.size, [0, 1], axis=0)

		# TODO: remove if not used
		# create two matrices of zeros which will be filled with values later on
		buffer = np.zeros((1, num_rows*num_cols), dtype="int").reshape(-1)

		# use the Matrix import to create buffer matrices with the given dimensions
		# multiple dimensions are needed to reduce in different steps properly
		# TODO: can be exchanged with a simple function creating a nested list based on the given inputs
		buffer64x64 = MatrixUtil.allocate_matrix(64, 64)
		buffer16x64 = MatrixUtil.allocate_matrix(16, 64)
		buffer16x16 = MatrixUtil.allocate_matrix(16, 16)

		# get a grayscaled and lumniated version of the image, this makes processing easier
		# also it reduces complexity for processing, since RGB images are harder to process
		buffer = self.fill_float_luma_from_buffer_image(img)

		# create hash value for the image
		hash_value = self.find_hash_256_from_float_luma(buffer, num_rows, num_cols, buffer64x64, buffer16x64, buffer16x16)
		return hash_value

	@staticmethod
	def fill_float_luma_from_buffer_image(img: Image) -> np.array:

		# convert the image object to numpy array based on RGB values. Numpy will contain float 16 values for each pixel
		# with R, G and B channels
		rgb_image = np.asarray(img.convert("RGB"), dtype="float16")

		# gray scale / illuminate the image
		return np.floor(np.ravel(np.dot(rgb_image[:][:], [0.299, 0.587, 0.114])))

	def find_hash_256_from_float_luma(self, full_buffer: list, num_rows: int, num_cols: int,
									  buffer64x64: list, buffer16x64: list, buffer16x16: list):

		# get the window size dimensions for the convolution process / window operation
		# it considers the given amount of columns and rows and creates an int based on the
		# size of 64 pixels
		window_size_cols = self.compute_box_filter_window_size(num_cols)
		window_size_rows = self.compute_box_filter_window_size(num_rows)

		# apply convolution 2d operation to reduce the image size to an average representation
		full_buffer = self.box_filter(full_buffer, num_rows, num_cols, window_size_rows, window_size_cols)

		# reduce the image to a 64x64 representation and use the buffer64x64 as array to be filled
		self.decimate_float(full_buffer, num_rows, num_cols, buffer64x64)

		# now go one step further and reduce the 64x64 pixel array to a 16x16 image array
		# the size has an influence on precision and used memory. So there is kind of a tradeoff between
		# computation cost and accuracy - 16x16 is a standard and often used in such a hashing search engine
		# play around with that if you need to.
		self.dct_64_to_16(buffer64x64, buffer16x64, buffer16x16)
		hash_value = self.dct_2_hash(buffer16x16)
		return hash_value

	@classmethod
	def compute_box_filter_window_size(cls, dimension: int) -> int:
		"""
		Create a standardized box size dimension based on a given amount of rows or columns.

		:param dimension: amount of rows or columns
		:return: integer which defines the box size of with the given window size divisor
		"""

		# Window Size divisor = 64
		return int(
			(dimension + cls.FIND_WINDOW_SIZE_DIVISOR - 1)
			/ cls.FIND_WINDOW_SIZE_DIVISOR
		)

	@classmethod
	def box_filter(cls, buffer: np.array, rows: int, cols: int, row_win: int, col_win: int) -> list:
		"""
		create a averaged convolved version of the image pixels. The idea is to make each pixel an average
		representation of it's neighbors.

		:param buffer: input array containing the image pixels in a 1d array
		:param rows: the amount of rows of the resized image
		:param cols: the amount of columns of the resized image
		:param row_win: amount of rows for the sliding window
		:param col_win: amount of columns for the sliding window

		:return returns the modified 1d image array as list with an averaged pixel representation
		"""

		# transform 1d array back to image 2d dimensions for window operation
		buffer = buffer.reshape(rows, cols)

		# apply window operation and transform back to 1d array and list type
		convolved_buffer = cls.convolve_avg(buffer, row_win, col_win)
		convolved_buffer = convolved_buffer.reshape(1, rows * cols)[0].tolist()

		return convolved_buffer

	@classmethod
	def decimate_float(cls, input_array: list, num_rows: int, num_cols: int, out: list) -> None:
		"""
		reduce the input array / list to 64x64 dimension in taking only a subset of pixels

		:param input_array: as nested list
		:param num_rows: amount of rows to consider
		:param num_cols: amount of columns to consider
		:param out: placeholder array as nested list which will be filled with sub samples of the input image

		:return None. Since this is a class method and out will be transformed due to pointing on the same memory
		"""

		# iterate over each pixel in row
		for i in np.arange(64):
			ini = int(np.true_divide(np.multiply((i + 0.5), num_rows), 64))

			# iterate over each pixel in column
			for j in np.arange(64):
				inj = int(np.true_divide(np.multiply((j + 0.5), num_cols), 64))

				# assign sub value of input array at index i, j of the output array
				out[i][j] = input_array[np.multiply(ini, num_cols) + inj]

	def dct_64_to_16(self, a, t, b):
		"""
		Full 64x64 to 64x64 can be optimized e.g. the Lee algorithm.
		But here we only want slots (1-16)x(1-16) of the full 64x64 output.
		Careful experiments showed that using Lee along all 64 slots in one
		dimension, then Lee along 16 slots in the second, followed by
		extracting slots 1-16 of the output, was actually slower than the
		current implementation which is completely non-clever/non-Lee but
		computes only what is needed.

		dct_64_to_16(buffer64x64, buffer16x64, buffer16x16)
		:param a: nested list array containing the input to down sample (64x64)
		:param t: nested list array placeholder for dimensions 16x64 - will be filled
		:param b:
		"""

		# Assign  dct matrix (see top of the script)
		d = self.dct_matrix

		# b = d a dt
		# b = (d a) dt ; t = d a
		# t is 16x64;

		# t = d a
		# tij = sum {k} dik akj

		t = [0] * 16
		for i in range(0, 16):
			ti = [0] * 64

			for j in range(0, 64):
				tij = 0.0

				for k in range(0, 64):
					tij += d[i][k] * a[k][j]
				ti[j] = tij
			t[i] = ti

		# B = T Dt
		# Bij = sum {k} Tik Djk
		for i in range(16):
			for j in range(16):
				sumk = float(0.0)
				for k in range(64):
					sumk += t[i][k] * d[j][k]

				# B is buffer16x16
				b[i][j] = sumk

	@staticmethod
	def dct_2_hash(dct16x16: list) -> ImageHash:
		"""
		binarizes dct func output & unravels into array
		Each bit of the 16x16 output hash is for whether the given frequency
		component is greater than the median frequency component or not.

		:param dct16x16: 2d array as nested list

		:return: ImageHash class which contains a hash value for the given image
		"""

		# create an empty placeholder array for the hash value
		hash_array = np.zeros((16, 16), dtype="int")

		# calculate the median values for the dct_matrix
		dct_median = MatrixUtil.torben(dct16x16, 16, 16)

		# iterate through each pixel value of in row and column
		for i in range(16):
			for j in range(16):

				# when the given pixel value is larger than the dct_median value then set the given i, j index to 1
				if dct16x16[i][j] > dct_median:
					hash_array[15-i, 15-j] = 1

		# make the 2d array to a 1d array and wrap it in the ImageHash class - for more insights check ImageHash docs
		return ImageHash(hash_array.reshape((256,)))

	@staticmethod
	def convolve_avg(x: np.array, row_win: int, col_win: int) -> np.array:
		"""
		compute convolution by configuring a kernel based on window height & width
		and under application of symmetric boundary conditions to anticipate edges at 
		image boundaries.

		:param x: input 2d array containing the image pixel values
		:param row_win: amount of rows for the sliding window (kernel)
		:param col_win: amount of columns for the sliding window (kernel)

		:return averaged pixel array
		"""

		# dynamize kernel size based on window height & width
		# creates a 2d window with the given dimensions containing ones - for computing the average values
		# without any weighting of the pixels / different to CONV2D Layers e.g. in tensorflow since the
		# convolution does not learn any filter
		kernel = np.ones((row_win, col_win))
		
		# compute convoluted image
		# has the same size as the input image and uses symmetric padding
		neighbor_sum = convolve2d(
			x, kernel, mode='same',
			boundary='symm')

		# compute divisor of same dimension for subsequent averaging
		num_neighbor = convolve2d(
			np.ones(x.shape), kernel, mode='same',
			boundary='symm')

		# divide the two arrays to get the average and return the resulting array
		return neighbor_sum / num_neighbor

	@classmethod
	def pretty_hash(cls, hash_value):
		"""
		turns list like image hash object into 16 by 16 matrix
		"""
		# hashes are 16x16. Print in this format
		assert len(hash_value.hash) == 256, "This function only works with 256-bit hashes"
		return np.array(hash_value.hash).astype(int).reshape((16, 16))
