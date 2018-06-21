import sys
import pandas as pd
import numpy as np 
import copy

if __name__ == '__main__':

	data = []

	data.append(pd.read_csv("./gga.csv",header=None))
	data.append(pd.read_csv("./cnn_gga.csv",header=None))
	data.append(pd.read_csv("./enn_gga.csv",header=None))
	data.append(pd.read_csv("./rss_gga.csv",header=None))
	data.append(pd.read_csv("./cnn_rss_gga.csv",header=None))
	data.append(pd.read_csv("./enn_rss_gga.csv",header=None))

	data.append(pd.read_csv("./ssga.csv",header=None))
	data.append(pd.read_csv("./cnn_ssga.csv",header=None))
	data.append(pd.read_csv("./enn_ssga.csv",header=None))
	data.append(pd.read_csv("./rss_ssga.csv",header=None))
	data.append(pd.read_csv("./cnn_rss_ssga.csv",header=None))
	data.append(pd.read_csv("./enn_rss_ssga.csv",header=None))

	data.append(pd.read_csv("./ma.csv",header=None))
	data.append(pd.read_csv("./cnn_ma.csv",header=None))
	data.append(pd.read_csv("./enn_ma.csv",header=None))
	data.append(pd.read_csv("./rss_ma.csv",header=None))
	data.append(pd.read_csv("./cnn_rss_ma.csv",header=None))
	data.append(pd.read_csv("./enn_rss_ma.csv",header=None))

	data.append(pd.read_csv("./chc.csv",header=None))
	data.append(pd.read_csv("./cnn_chc.csv",header=None))
	data.append(pd.read_csv("./enn_chc.csv",header=None))
	data.append(pd.read_csv("./rss_chc.csv",header=None))
	data.append(pd.read_csv("./cnn_rss_chc.csv",header=None))
	data.append(pd.read_csv("./enn_rss_chc.csv",header=None))
