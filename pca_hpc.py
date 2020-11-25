from pandas import read_csv, DataFrame
import numpy as np
from numba import njit, jit


# @njit(parallel=True, cache=True)
def normalize(data):
	mu = np.mean(data)
	sigma = np.std(data)
	return ((data - mu), mu, sigma)


# @jit(parallel=True, forceobj=True)
def normalize_matrix(data_matrix):
	normal_matrix = []
	meta = []
	data_matrix = data_matrix.T
	for x in range(0, data_matrix.shape[0]):
		temp = normalize(data_matrix[x])
		normal_matrix.append(temp[0])
		meta.append(temp[1:])
	return np.array(normal_matrix).T, meta


# @njit()
def eigen_decomp(cov_frame):
	return np.linalg.eig(cov_frame)


# @jit(parallel=True, forceobj=True)
def frame_cov(m, *y):
	frame = {0:m}
	for x in range(1, len(y)+1):
		frame[x] = y[x-1]
	frame = DataFrame.from_dict(frame).cov().to_numpy()
	return frame


# @jit(parallel=True, forceobj=True)
def pca_node(data_1, *data_x):
	data = (data_1, *data_x)
	data = normalize_matrix(np.array(data).T)
	stat = {}
	for i in range(0, len(data[1])):
		stat[i] = data[1][i]
	matrix_ = np.array(data[0].T)
	data = tuple(matrix_)
	cov_ = frame_cov(data[0], *data[1:])
	(eigen_value, eigen_vector) = np.linalg.eig(cov_)
	matrix_ = eigen_vector.T.dot(matrix_)
	return (matrix_, eigen_value, eigen_vector, cov_, stat)


def decider_node(pca_data):
	trace = np.trace(pca_data[3])
	if (pca_data[3][0][0]) / trace >= (pca_data[3][1][1]) / trace:
		return (pca_data[0][0],
				pca_data[1][0],
				pca_data[2][0], 
				(pca_data[3][0][0]) / trace, 
				pca_data[3],
				pca_data[4])
	return (pca_data[0][1],
			pca_data[1][1],
			pca_data[2][1],
			(pca_data[3][1][1]) / trace,
			pca_data[3],
			pca_data[4])


def compressor_node(data_1, *data_x):
	pca_data = pca_node(data_1, *data_x)
	return decider_node(pca_data)


# @jit(parallel=True, forceobj=True)
def compressor_layer(matrix, cov_matrix):
	drop = (0, 0)
	shape = cov_matrix.shape
	cov_trace = np.trace(cov_matrix)
	best = (0, 0, 0, 0, cov_matrix)
	if shape[0] == 2:
		best = compressor_node(matrix[0], matrix[1])
		drop = (0, 1)
		return best, drop, np.trace(best[4]) / cov_trace
	for i in range(0, shape[0]):
		for j in range(i+1, shape[0]):
			temp = compressor_node(matrix[i], matrix[j])
			if np.trace(best[4]) / cov_trace > (np.trace(temp[4])) / cov_trace:
				best = temp
				drop = (i, j)
	return best, drop, np.trace(best[4]) / cov_trace


def compressor_network(frame, target_dim=1):
	if target_dim < 1:
		raise ValueError('Target Dimension can\'t be less than 1')
	pca_meta = {}
	cols = frame.columns
	count = len(cols) - 1
	while len(frame.columns) > target_dim:
		pca_meta[count] = {}
		stage, drop, syn = compressor_layer(frame.to_numpy().T, frame.cov().to_numpy())
		drop_cols = frame.columns[list(drop)]
		frame = frame.drop(drop_cols, axis=1)
		frame[f'({drop_cols[0]}, {drop_cols[1]})'] = stage[0]
		pca_meta[count]['eigen_value'] = stage[1]
		pca_meta[count]['eigen_vector'] = stage[2]
		pca_meta[count]['trace'] = stage[3]
		pca_meta[count]['cov_matrix'] = stage[4]
		pca_meta[count]['stats'] = stage[5]
		pca_meta[count]['drop'] = drop
		pca_meta[count]['syn'] = syn
		count -= 1
	return frame, DataFrame.from_dict(pca_meta).T