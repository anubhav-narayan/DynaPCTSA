import pytest

from pctsa import pca_hpc
from pandas import read_csv, DataFrame
import numpy as np

def test_load_data():
	'''
	Data Loading Test
	'''
	LOAD_DATA = read_csv('./test_data/pm25_x.csv')
	assert len(LOAD_DATA.columns) == 10

def test_clean_data():
	'''
	Data Cleaning Test
	'''
	bpi_ts = read_csv('./test_data/pm25_x.csv')
	bpi_ts = bpi_ts[['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws_x']]
	bpi_ts = bpi_ts.fillna(method='bfill')
	
	assert np.nan not in bpi_ts

def test_normalize():
	'''
	Tests normalize
	'''
	bpi_ts = read_csv('./test_data/pm25_x.csv')
	bpi_ts = bpi_ts['pm2.5']
	bpi_ts = bpi_ts.fillna(method='bfill')
	bpi_ts = pca_hpc.normalize(bpi_ts.to_numpy())

	assert bpi_ts[0].all() <= 1 and bpi_ts[0].all() >= 0


def test_normalize_matrix():
	'''
	Tests normalize_matrix
	'''
	bpi_ts = read_csv('./test_data/pm25_x.csv')
	bpi_ts = bpi_ts[['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws_x']]
	bpi_ts = bpi_ts.fillna(method='bfill')
	bpi_ts = pca_hpc.normalize_matrix(bpi_ts.to_numpy())

	assert bpi_ts[0].all() <= 1 and bpi_ts[0].all() >= 0


def test_frame_cov():
	'''
	Tests frame_cov
	'''
	res = pca_hpc.frame_cov([1, 1], [1, 1])
	assert res.all() == 0


def test_compressor_node():
	'''
	Tests compressor_node
	'''
	bpi_ts = read_csv('./test_data/pm25_x.csv')
	bpi_ts = bpi_ts[['pm2.5', 'DEWP']]
	bpi_ts = bpi_ts.fillna(method='bfill')
	bpi_ts = DataFrame(pca_hpc.normalize_matrix(bpi_ts.to_numpy())[0], columns=bpi_ts.columns)
	temp = pca_hpc.compressor_node(bpi_ts['pm2.5'].to_numpy(), bpi_ts['DEWP'].to_numpy())
	print(temp)
	assert temp[1] == pytest.approx(8362.166621390654)


def test_compressor_layer():
	'''
	Tests compressor_layer
	'''
	bpi_ts = read_csv('./test_data/pm25_x.csv', engine='c', low_memory=True)
	bpi_ts = bpi_ts[['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws_x']]
	bpi_ts = bpi_ts.fillna(method='bfill')
	x = DataFrame(pca_hpc.normalize_matrix(bpi_ts.to_numpy())[0], columns=bpi_ts.columns)
	temp = pca_hpc.compressor_layer(x.to_numpy().T, x.cov().to_numpy())
	print(temp)
	assert temp[1] == (3, 4) or temp[4] == pytest.approx(0.012578128797346716)


def test_compressor_network_fail():
	'''
	Tests compressor_network
	'''
	try:
		bpi_ts = read_csv('./test_data/pm25_x.csv', engine='c', low_memory=True)
		x = pca_hpc.compressor_network(bpi_ts, 0)
	except Exception as e:
		assert type(e) is type(ValueError('Target Dimension can\'t be less than 1'))



def test_compressor_network_5_2():
	'''
	Tests compressor_network
	'''
	bpi_ts = read_csv('./test_data/pm25_x.csv', engine='c', low_memory=True)
	bpi_ts = bpi_ts[['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws_x']]
	bpi_ts = bpi_ts.fillna(method='bfill')
	y = DataFrame(pca_hpc.normalize_matrix(bpi_ts.to_numpy())[0], columns=bpi_ts.columns)
	x = pca_hpc.compressor_network(y, 2)
	trace = np.trace(x[0].cov()) / np.trace(y.cov())
	print(trace, np.sum(np.diag(y.cov())[:2]) / np.trace(y.cov()))
	assert trace >= np.sum(np.diag(y.cov())[:2]) / np.trace(y.cov())


def test_compressor_network_5_3():
	'''
	Tests compressor_network
	'''
	bpi_ts = read_csv('./test_data/pm25_x.csv', engine='c', low_memory=True)
	bpi_ts = bpi_ts[['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws_x']]
	bpi_ts = bpi_ts.fillna(method='bfill')
	y = DataFrame(pca_hpc.normalize_matrix(bpi_ts.to_numpy())[0], columns=bpi_ts.columns)
	x = pca_hpc.compressor_network(y, 3)
	trace = np.trace(x[0].cov()) / np.trace(y.cov())
	print(trace, np.sum(np.diag(y.cov())[:3]) / np.trace(y.cov()))
	assert trace >= np.sum(np.diag(y.cov())[:3]) / np.trace(y.cov())


def test_compressor_network_5_4():
	'''
	Tests compressor_network
	'''
	bpi_ts = read_csv('./test_data/pm25_x.csv', engine='c', low_memory=True)
	bpi_ts = bpi_ts[['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws_x']]
	bpi_ts = bpi_ts.fillna(method='bfill')
	y = DataFrame(pca_hpc.normalize_matrix(bpi_ts.to_numpy())[0], columns=bpi_ts.columns)
	x = pca_hpc.compressor_network(y, 4)
	trace = np.trace(x[0].cov()) / np.trace(y.cov())
	print(trace, np.sum(np.diag(y.cov())[:4]) / np.trace(y.cov()))
	assert trace >= np.sum(np.diag(y.cov())[:4]) / np.trace(y.cov())


def test_compressor_network_stress_20_5():
	'''
	Tests compressor_network
	'''
	bpi_ts = read_csv('./test_data/synthetic_20.csv', engine='c', low_memory=True)
	y = DataFrame(pca_hpc.normalize_matrix(bpi_ts.to_numpy())[0], columns=bpi_ts.columns)
	x = pca_hpc.compressor_network(y, 5)
	trace = np.trace(x[0].cov()) / np.trace(y.cov())
	print(trace, np.sum(np.diag(y.cov())[:5]) / np.trace(y.cov()))
	assert trace >= np.sum(np.diag(y.cov())[:5]) / np.trace(y.cov())


def test_compressor_network_stress_20_1():
	'''
	Tests compressor_network
	'''
	bpi_ts = read_csv('./test_data/synthetic_20.csv', engine='c', low_memory=True)
	y = DataFrame(pca_hpc.normalize_matrix(bpi_ts.to_numpy())[0], columns=bpi_ts.columns)
	x = pca_hpc.compressor_network(y, 1)
	trace = np.trace(x[0].cov()) / np.trace(y.cov())
	print(trace, np.sum(np.diag(y.cov())[:1]) / np.trace(y.cov()))
	assert trace >= np.sum(np.diag(y.cov())[:1]) / np.trace(y.cov())


def test_compressor_network_stress_50_5():
	'''
	Tests compressor_network
	'''
	bpi_ts = read_csv('./test_data/synthetic_50.csv', engine='c', low_memory=True)
	bpi_ts = bpi_ts.drop([0])
	y = DataFrame(pca_hpc.normalize_matrix(bpi_ts.to_numpy())[0], columns=bpi_ts.columns)
	x = pca_hpc.compressor_network(y, 5)
	trace = np.trace(x[0].cov()) / np.trace(y.cov())
	print(trace, np.sum(np.diag(y.cov())[:5]) / np.trace(y.cov()))
	assert trace >= np.sum(np.diag(y.cov())[:5]) / np.trace(y.cov())


def test_compressor_network_stress_50_5_2():
	'''
	Tests compressor_network
	'''
	bpi_ts = read_csv('./test_data/synthetic_50_2.csv', engine='c', low_memory=True)
	bpi_ts = bpi_ts.drop([0])
	y = DataFrame(pca_hpc.normalize_matrix(bpi_ts.to_numpy())[0], columns=bpi_ts.columns)
	x = pca_hpc.compressor_network(y, 5)
	trace = np.trace(x[0].cov()) / np.trace(y.cov())
	print(trace, np.sum(np.diag(y.cov())[:5]) / np.trace(y.cov()))
	assert trace >= np.sum(np.diag(y.cov())[:5]) / np.trace(y.cov())