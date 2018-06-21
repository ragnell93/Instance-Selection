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

	new_data = []

	for i in range(0,45):
		new_data.append(pd.DataFrame())
		for j in range(0,24):
			new_data[i] = new_data[i].append(data[j].iloc[i,:],ignore_index=True)


	accuracy = []
	for i in range(0,45):
		accuracy.append(new_data[i].sort_values(by=2,ascending=False).reset_index(drop=True))

	kappa = []
	for i in range(0,45):
		kappa.append(new_data[i].sort_values(by=4,ascending=False).reset_index(drop=True))

	reduction = []
	for i in range(0,45):
		reduction.append(new_data[i].sort_values(by=5,ascending=False).reset_index(drop=True))

	time = []
	for i in range(0,45):
		time.append(new_data[i].sort_values(by=6).reset_index(drop=True))

	acc_red = []
	for i in range(0,45):
		acc_red.append(new_data[i].sort_values(by=7,ascending=False).reset_index(drop=True))

	kap_red = []
	for i in range(0,45):
		kap_red.append(new_data[i].sort_values(by=8,ascending=False).reset_index(drop=True))




	small_acc = [0] * 24
	first_small_acc = [0] * 24
	medium_acc = [0] * 24
	first_medium_acc = [0] * 24
	large_acc = [0] * 24
	first_large_acc = [0] * 24



	for i in range(0,26):
		small_acc[0] += accuracy[i][accuracy[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[0] += 1
	small_acc[0] = small_acc[0]/26

	for i in range(0,26):
		small_acc[1] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[1] += 1
	small_acc[1] = small_acc[1]/26

	for i in range(0,26):
		small_acc[2] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[2] += 1
	small_acc[2] = small_acc[2]/26

	for i in range(0,26):
		small_acc[3] += accuracy[i][accuracy[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[3] += 1
	small_acc[3] = small_acc[3]/26

	for i in range(0,26):
		small_acc[4] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[4] += 1
	small_acc[4] = small_acc[4]/26

	for i in range(0,26):
		small_acc[5] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[5] += 1
	small_acc[5] = small_acc[5]/26



	for i in range(0,26):
		small_acc[6] += accuracy[i][accuracy[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[6] += 1
	small_acc[6] = small_acc[6]/26

	for i in range(0,26):
		small_acc[7] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[7] += 1
	small_acc[7] = small_acc[7]/26

	for i in range(0,26):
		small_acc[8] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[8] += 1
	small_acc[8] = small_acc[8]/26

	for i in range(0,26):
		small_acc[9] += accuracy[i][accuracy[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[9] += 1
	small_acc[9] = small_acc[9]/26

	for i in range(0,26):
		small_acc[10] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[10] += 1
	small_acc[10] = small_acc[10]/26

	for i in range(0,26):
		small_acc[11] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[11] += 1
	small_acc[11] = small_acc[11]/26

	

	for i in range(0,26):
		small_acc[12] += accuracy[i][accuracy[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[12] += 1
	small_acc[12] = small_acc[12]/26

	for i in range(0,26):
		small_acc[13] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[13] += 1
	small_acc[13] = small_acc[13]/26

	for i in range(0,26):
		small_acc[14] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[14] += 1
	small_acc[14] = small_acc[14]/26

	for i in range(0,26):
		small_acc[15] += accuracy[i][accuracy[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[15] += 1
	small_acc[15] = small_acc[15]/26

	for i in range(0,26):
		small_acc[16] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[16] += 1
	small_acc[16] = small_acc[16]/26

	for i in range(0,26):
		small_acc[17] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[17] += 1
	small_acc[17] = small_acc[17]/26



	for i in range(0,26):
		small_acc[18] += accuracy[i][accuracy[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[18] += 1
	small_acc[18] = small_acc[18]/26

	for i in range(0,26):
		small_acc[19] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[19] += 1
	small_acc[19] = small_acc[19]/26

	for i in range(0,26):
		small_acc[20] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[20] += 1
	small_acc[20] = small_acc[20]/26

	for i in range(0,26):
		small_acc[21] += accuracy[i][accuracy[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[21] += 1
	small_acc[21] = small_acc[21]/26

	for i in range(0,26):
		small_acc[22] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[22] += 1
	small_acc[22] = small_acc[22]/26

	for i in range(0,26):
		small_acc[23] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_acc[23] += 1
	small_acc[23] = small_acc[23]/26





	for i in range(26,43):
		medium_acc[0] += accuracy[i][accuracy[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[0] += 1
	medium_acc[0] = medium_acc[0]/17

	for i in range(26,43):
		medium_acc[1] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[1] += 1
	medium_acc[1] = medium_acc[1]/17

	for i in range(26,43):
		medium_acc[2] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[2] += 1
	medium_acc[2] = medium_acc[2]/17

	for i in range(26,43):
		medium_acc[3] += accuracy[i][accuracy[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[3] += 1
	medium_acc[3] = medium_acc[3]/17

	for i in range(26,43):
		medium_acc[4] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[4] += 1
	medium_acc[4] = medium_acc[4]/17

	for i in range(26,43):
		medium_acc[5] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[5] += 1
	medium_acc[5] = medium_acc[5]/17



	for i in range(26,43):
		medium_acc[6] += accuracy[i][accuracy[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[6] += 1
	medium_acc[6] = medium_acc[6]/17

	for i in range(26,43):
		medium_acc[7] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[7] += 1
	medium_acc[7] = medium_acc[7]/17

	for i in range(26,43):
		medium_acc[8] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[8] += 1
	medium_acc[8] = medium_acc[8]/17

	for i in range(26,43):
		medium_acc[9] += accuracy[i][accuracy[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[9] += 1
	medium_acc[9] = medium_acc[9]/17

	for i in range(26,43):
		medium_acc[10] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[10] += 1
	medium_acc[10] = medium_acc[10]/17

	for i in range(26,43):
		medium_acc[11] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[11] += 1
	medium_acc[11] = medium_acc[11]/17

	

	for i in range(26,43):
		medium_acc[12] += accuracy[i][accuracy[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[12] += 1
	medium_acc[12] = medium_acc[12]/17

	for i in range(26,43):
		medium_acc[13] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[13] += 1
	medium_acc[13] = medium_acc[13]/17

	for i in range(26,43):
		medium_acc[14] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[14] += 1
	medium_acc[14] = medium_acc[14]/17

	for i in range(26,43):
		medium_acc[15] += accuracy[i][accuracy[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[15] += 1
	medium_acc[15] = medium_acc[15]/17

	for i in range(26,43):
		medium_acc[16] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[16] += 1
	medium_acc[16] = medium_acc[16]/17

	for i in range(26,43):
		medium_acc[17] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[17] += 1
	medium_acc[17] = medium_acc[17]/17



	for i in range(26,43):
		medium_acc[18] += accuracy[i][accuracy[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[18] += 1
	medium_acc[18] = medium_acc[18]/17

	for i in range(26,43):
		medium_acc[19] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[19] += 1
	medium_acc[19] = medium_acc[19]/17

	for i in range(26,43):
		medium_acc[20] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[20] += 1
	medium_acc[20] = medium_acc[20]/17

	for i in range(26,43):
		medium_acc[21] += accuracy[i][accuracy[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[21] += 1
	medium_acc[21] = medium_acc[21]/17

	for i in range(26,43):
		medium_acc[22] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[22] += 1
	medium_acc[22] = medium_acc[22]/17

	for i in range(26,43):
		medium_acc[23] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_acc[23] += 1
	medium_acc[23] = medium_acc[23]/17








	for i in range(43,45):
		large_acc[0] += accuracy[i][accuracy[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[0] += 1
	large_acc[0] = large_acc[0]/2

	for i in range(43,45):
		large_acc[1] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[1] += 1
	large_acc[1] = large_acc[1]/2

	for i in range(43,45):
		large_acc[2] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[2] += 1
	large_acc[2] = large_acc[2]/2

	for i in range(43,45):
		large_acc[3] += accuracy[i][accuracy[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[3] += 1
	large_acc[3] = large_acc[3]/2

	for i in range(43,45):
		large_acc[4] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[4] += 1
	large_acc[4] = large_acc[4]/2

	for i in range(43,45):
		large_acc[5] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[5] += 1
	large_acc[5] = large_acc[5]/2



	for i in range(43,45):
		large_acc[6] += accuracy[i][accuracy[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[6] += 1
	large_acc[6] = large_acc[6]/2

	for i in range(43,45):
		large_acc[7] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[7] += 1
	large_acc[7] = large_acc[7]/2

	for i in range(43,45):
		large_acc[8] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[8] += 1
	large_acc[8] = large_acc[8]/2

	for i in range(43,45):
		large_acc[9] += accuracy[i][accuracy[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[9] += 1
	large_acc[9] = large_acc[9]/2

	for i in range(43,45):
		large_acc[10] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[10] += 1
	large_acc[10] = large_acc[10]/2

	for i in range(43,45):
		large_acc[11] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[11] += 1
	large_acc[11] = large_acc[11]/2

	

	for i in range(43,45):
		large_acc[12] += accuracy[i][accuracy[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[12] += 1
	large_acc[12] = large_acc[12]/2

	for i in range(43,45):
		large_acc[13] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[13] += 1
	large_acc[13] = large_acc[13]/2

	for i in range(43,45):
		large_acc[14] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[14] += 1
	large_acc[14] = large_acc[14]/2

	for i in range(43,45):
		large_acc[15] += accuracy[i][accuracy[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[15] += 1
	large_acc[15] = large_acc[15]/2

	for i in range(43,45):
		large_acc[16] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[16] += 1
	large_acc[16] = large_acc[16]/2

	for i in range(43,45):
		large_acc[17] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[17] += 1
	large_acc[17] = large_acc[17]/2



	for i in range(43,45):
		large_acc[18] += accuracy[i][accuracy[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[18] += 1
	large_acc[18] = large_acc[18]/2

	for i in range(43,45):
		large_acc[19] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[19] += 1
	large_acc[19] = large_acc[19]/2

	for i in range(43,45):
		large_acc[20] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[20] += 1
	large_acc[20] = large_acc[20]/2

	for i in range(43,45):
		large_acc[21] += accuracy[i][accuracy[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[21] += 1
	large_acc[21] = large_acc[21]/2

	for i in range(43,45):
		large_acc[22] += accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[22] += 1
	large_acc[22] = large_acc[22]/2

	for i in range(43,45):
		large_acc[23] += accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((accuracy[i][accuracy[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_acc[23] += 1
	large_acc[23] = large_acc[23]/2


	print("accuracy")
	print("*"*100)
	print(small_acc)
	print(medium_acc)
	print(large_acc)


	print(first_small_acc)
	print(first_medium_acc)
	print(first_large_acc)

	print("*"*100)


	small_kap = [0] * 24
	first_small_kap = [0] * 24
	medium_kap = [0] * 24
	first_medium_kap = [0] * 24
	large_kap = [0] * 24
	first_large_kap = [0] * 24



	for i in range(0,26):
		small_kap[0] += kappa[i][kappa[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[0] += 1
	small_kap[0] = small_kap[0]/26

	for i in range(0,26):
		small_kap[1] += kappa[i][kappa[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[1] += 1
	small_kap[1] = small_kap[1]/26

	for i in range(0,26):
		small_kap[2] += kappa[i][kappa[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[2] += 1
	small_kap[2] = small_kap[2]/26

	for i in range(0,26):
		small_kap[3] += kappa[i][kappa[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[3] += 1
	small_kap[3] = small_kap[3]/26

	for i in range(0,26):
		small_kap[4] += kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[4] += 1
	small_kap[4] = small_kap[4]/26

	for i in range(0,26):
		small_kap[5] += kappa[i][kappa[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[5] += 1
	small_kap[5] = small_kap[5]/26



	for i in range(0,26):
		small_kap[6] += kappa[i][kappa[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[6] += 1
	small_kap[6] = small_kap[6]/26

	for i in range(0,26):
		small_kap[7] += kappa[i][kappa[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[7] += 1
	small_kap[7] = small_kap[7]/26

	for i in range(0,26):
		small_kap[8] += kappa[i][kappa[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[8] += 1
	small_kap[8] = small_kap[8]/26

	for i in range(0,26):
		small_kap[9] += kappa[i][kappa[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[9] += 1
	small_kap[9] = small_kap[9]/26

	for i in range(0,26):
		small_kap[10] += kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[10] += 1
	small_kap[10] = small_kap[10]/26

	for i in range(0,26):
		small_kap[11] += kappa[i][kappa[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[11] += 1
	small_kap[11] = small_kap[11]/26

	

	for i in range(0,26):
		small_kap[12] += kappa[i][kappa[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[12] += 1
	small_kap[12] = small_kap[12]/26

	for i in range(0,26):
		small_kap[13] += kappa[i][kappa[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[13] += 1
	small_kap[13] = small_kap[13]/26

	for i in range(0,26):
		small_kap[14] += kappa[i][kappa[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[14] += 1
	small_kap[14] = small_kap[14]/26

	for i in range(0,26):
		small_kap[15] += kappa[i][kappa[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[15] += 1
	small_kap[15] = small_kap[15]/26

	for i in range(0,26):
		small_kap[16] += kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[16] += 1
	small_kap[16] = small_kap[16]/26

	for i in range(0,26):
		small_kap[17] += kappa[i][kappa[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[17] += 1
	small_kap[17] = small_kap[17]/26



	for i in range(0,26):
		small_kap[18] += kappa[i][kappa[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[18] += 1
	small_kap[18] = small_kap[18]/26

	for i in range(0,26):
		small_kap[19] += kappa[i][kappa[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[19] += 1
	small_kap[19] = small_kap[19]/26

	for i in range(0,26):
		small_kap[20] += kappa[i][kappa[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[20] += 1
	small_kap[20] = small_kap[20]/26

	for i in range(0,26):
		small_kap[21] += kappa[i][kappa[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[21] += 1
	small_kap[21] = small_kap[21]/26

	for i in range(0,26):
		small_kap[22] += kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[22] += 1
	small_kap[22] = small_kap[22]/26

	for i in range(0,26):
		small_kap[23] += kappa[i][kappa[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_kap[23] += 1
	small_kap[23] = small_kap[23]/26





	for i in range(26,43):
		medium_kap[0] += kappa[i][kappa[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[0] += 1
	medium_kap[0] = medium_kap[0]/17

	for i in range(26,43):
		medium_kap[1] += kappa[i][kappa[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[1] += 1
	medium_kap[1] = medium_kap[1]/17

	for i in range(26,43):
		medium_kap[2] += kappa[i][kappa[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[2] += 1
	medium_kap[2] = medium_kap[2]/17

	for i in range(26,43):
		medium_kap[3] += kappa[i][kappa[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[3] += 1
	medium_kap[3] = medium_kap[3]/17

	for i in range(26,43):
		medium_kap[4] += kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[4] += 1
	medium_kap[4] = medium_kap[4]/17

	for i in range(26,43):
		medium_kap[5] += kappa[i][kappa[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[5] += 1
	medium_kap[5] = medium_kap[5]/17



	for i in range(26,43):
		medium_kap[6] += kappa[i][kappa[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[6] += 1
	medium_kap[6] = medium_kap[6]/17

	for i in range(26,43):
		medium_kap[7] += kappa[i][kappa[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[7] += 1
	medium_kap[7] = medium_kap[7]/17

	for i in range(26,43):
		medium_kap[8] += kappa[i][kappa[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[8] += 1
	medium_kap[8] = medium_kap[8]/17

	for i in range(26,43):
		medium_kap[9] += kappa[i][kappa[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[9] += 1
	medium_kap[9] = medium_kap[9]/17

	for i in range(26,43):
		medium_kap[10] += kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[10] += 1
	medium_kap[10] = medium_kap[10]/17

	for i in range(26,43):
		medium_kap[11] += kappa[i][kappa[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[11] += 1
	medium_kap[11] = medium_kap[11]/17

	

	for i in range(26,43):
		medium_kap[12] += kappa[i][kappa[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[12] += 1
	medium_kap[12] = medium_kap[12]/17

	for i in range(26,43):
		medium_kap[13] += kappa[i][kappa[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[13] += 1
	medium_kap[13] = medium_kap[13]/17

	for i in range(26,43):
		medium_kap[14] += kappa[i][kappa[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[14] += 1
	medium_kap[14] = medium_kap[14]/17

	for i in range(26,43):
		medium_kap[15] += kappa[i][kappa[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[15] += 1
	medium_kap[15] = medium_kap[15]/17

	for i in range(26,43):
		medium_kap[16] += kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[16] += 1
	medium_kap[16] = medium_kap[16]/17

	for i in range(26,43):
		medium_kap[17] += kappa[i][kappa[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[17] += 1
	medium_kap[17] = medium_kap[17]/17



	for i in range(26,43):
		medium_kap[18] += kappa[i][kappa[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[18] += 1
	medium_kap[18] = medium_kap[18]/17

	for i in range(26,43):
		medium_kap[19] += kappa[i][kappa[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[19] += 1
	medium_kap[19] = medium_kap[19]/17

	for i in range(26,43):
		medium_kap[20] += kappa[i][kappa[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[20] += 1
	medium_kap[20] = medium_kap[20]/17

	for i in range(26,43):
		medium_kap[21] += kappa[i][kappa[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[21] += 1
	medium_kap[21] = medium_kap[21]/17

	for i in range(26,43):
		medium_kap[22] += kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[22] += 1
	medium_kap[22] = medium_kap[22]/17

	for i in range(26,43):
		medium_kap[23] += kappa[i][kappa[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kap[23] += 1
	medium_kap[23] = medium_kap[23]/17








	for i in range(43,45):
		large_kap[0] += kappa[i][kappa[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[0] += 1
	large_kap[0] = large_kap[0]/2

	for i in range(43,45):
		large_kap[1] += kappa[i][kappa[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[1] += 1
	large_kap[1] = large_kap[1]/2

	for i in range(43,45):
		large_kap[2] += kappa[i][kappa[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[2] += 1
	large_kap[2] = large_kap[2]/2

	for i in range(43,45):
		large_kap[3] += kappa[i][kappa[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[3] += 1
	large_kap[3] = large_kap[3]/2

	for i in range(43,45):
		large_kap[4] += kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[4] += 1
	large_kap[4] = large_kap[4]/2

	for i in range(43,45):
		large_kap[5] += kappa[i][kappa[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[5] += 1
	large_kap[5] = large_kap[5]/2



	for i in range(43,45):
		large_kap[6] += kappa[i][kappa[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[6] += 1
	large_kap[6] = large_kap[6]/2

	for i in range(43,45):
		large_kap[7] += kappa[i][kappa[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[7] += 1
	large_kap[7] = large_kap[7]/2

	for i in range(43,45):
		large_kap[8] += kappa[i][kappa[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[8] += 1
	large_kap[8] = large_kap[8]/2

	for i in range(43,45):
		large_kap[9] += kappa[i][kappa[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[9] += 1
	large_kap[9] = large_kap[9]/2

	for i in range(43,45):
		large_kap[10] += kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[10] += 1
	large_kap[10] = large_kap[10]/2

	for i in range(43,45):
		large_kap[11] += kappa[i][kappa[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[11] += 1
	large_kap[11] = large_kap[11]/2

	

	for i in range(43,45):
		large_kap[12] += kappa[i][kappa[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[12] += 1
	large_kap[12] = large_kap[12]/2

	for i in range(43,45):
		large_kap[13] += kappa[i][kappa[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[13] += 1
	large_kap[13] = large_kap[13]/2

	for i in range(43,45):
		large_kap[14] += kappa[i][kappa[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[14] += 1
	large_kap[14] = large_kap[14]/2

	for i in range(43,45):
		large_kap[15] += kappa[i][kappa[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[15] += 1
	large_kap[15] = large_kap[15]/2

	for i in range(43,45):
		large_kap[16] += kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[16] += 1
	large_kap[16] = large_kap[16]/2

	for i in range(43,45):
		large_kap[17] += kappa[i][kappa[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[17] += 1
	large_kap[17] = large_kap[17]/2



	for i in range(43,45):
		large_kap[18] += kappa[i][kappa[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[18] += 1
	large_kap[18] = large_kap[18]/2

	for i in range(43,45):
		large_kap[19] += kappa[i][kappa[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[19] += 1
	large_kap[19] = large_kap[19]/2

	for i in range(43,45):
		large_kap[20] += kappa[i][kappa[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[20] += 1
	large_kap[20] = large_kap[20]/2

	for i in range(43,45):
		large_kap[21] += kappa[i][kappa[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[21] += 1
	large_kap[21] = large_kap[21]/2

	for i in range(43,45):
		large_kap[22] += kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[22] += 1
	large_kap[22] = large_kap[22]/2

	for i in range(43,45):
		large_kap[23] += kappa[i][kappa[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((kappa[i][kappa[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_kap[23] += 1
	large_kap[23] = large_kap[23]/2


	print("kappa")
	print("*"*100)
	print(small_kap)
	print(medium_kap)
	print(large_kap)


	print(first_small_kap)
	print(first_medium_kap)
	print(first_large_kap)

	print("*"*100)


	small_red = [0] * 24
	first_small_red = [0] * 24
	medium_red = [0] * 24
	first_medium_red = [0] * 24
	large_red = [0] * 24
	first_large_red = [0] * 24



	for i in range(0,26):
		small_red[0] += reduction[i][reduction[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[0] += 1
	small_red[0] = small_red[0]/26

	for i in range(0,26):
		small_red[1] += reduction[i][reduction[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[1] += 1
	small_red[1] = small_red[1]/26

	for i in range(0,26):
		small_red[2] += reduction[i][reduction[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[2] += 1
	small_red[2] = small_red[2]/26

	for i in range(0,26):
		small_red[3] += reduction[i][reduction[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[3] += 1
	small_red[3] = small_red[3]/26

	for i in range(0,26):
		small_red[4] += reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[4] += 1
	small_red[4] = small_red[4]/26

	for i in range(0,26):
		small_red[5] += reduction[i][reduction[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[5] += 1
	small_red[5] = small_red[5]/26



	for i in range(0,26):
		small_red[6] += reduction[i][reduction[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[6] += 1
	small_red[6] = small_red[6]/26

	for i in range(0,26):
		small_red[7] += reduction[i][reduction[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[7] += 1
	small_red[7] = small_red[7]/26

	for i in range(0,26):
		small_red[8] += reduction[i][reduction[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[8] += 1
	small_red[8] = small_red[8]/26

	for i in range(0,26):
		small_red[9] += reduction[i][reduction[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[9] += 1
	small_red[9] = small_red[9]/26

	for i in range(0,26):
		small_red[10] += reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[10] += 1
	small_red[10] = small_red[10]/26

	for i in range(0,26):
		small_red[11] += reduction[i][reduction[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[11] += 1
	small_red[11] = small_red[11]/26

	

	for i in range(0,26):
		small_red[12] += reduction[i][reduction[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[12] += 1
	small_red[12] = small_red[12]/26

	for i in range(0,26):
		small_red[13] += reduction[i][reduction[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[13] += 1
	small_red[13] = small_red[13]/26

	for i in range(0,26):
		small_red[14] += reduction[i][reduction[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[14] += 1
	small_red[14] = small_red[14]/26

	for i in range(0,26):
		small_red[15] += reduction[i][reduction[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[15] += 1
	small_red[15] = small_red[15]/26

	for i in range(0,26):
		small_red[16] += reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[16] += 1
	small_red[16] = small_red[16]/26

	for i in range(0,26):
		small_red[17] += reduction[i][reduction[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[17] += 1
	small_red[17] = small_red[17]/26



	for i in range(0,26):
		small_red[18] += reduction[i][reduction[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[18] += 1
	small_red[18] = small_red[18]/26

	for i in range(0,26):
		small_red[19] += reduction[i][reduction[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[19] += 1
	small_red[19] = small_red[19]/26

	for i in range(0,26):
		small_red[20] += reduction[i][reduction[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[20] += 1
	small_red[20] = small_red[20]/26

	for i in range(0,26):
		small_red[21] += reduction[i][reduction[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[21] += 1
	small_red[21] = small_red[21]/26

	for i in range(0,26):
		small_red[22] += reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[22] += 1
	small_red[22] = small_red[22]/26

	for i in range(0,26):
		small_red[23] += reduction[i][reduction[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_red[23] += 1
	small_red[23] = small_red[23]/26





	for i in range(26,43):
		medium_red[0] += reduction[i][reduction[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[0] += 1
	medium_red[0] = medium_red[0]/17

	for i in range(26,43):
		medium_red[1] += reduction[i][reduction[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[1] += 1
	medium_red[1] = medium_red[1]/17

	for i in range(26,43):
		medium_red[2] += reduction[i][reduction[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[2] += 1
	medium_red[2] = medium_red[2]/17

	for i in range(26,43):
		medium_red[3] += reduction[i][reduction[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[3] += 1
	medium_red[3] = medium_red[3]/17

	for i in range(26,43):
		medium_red[4] += reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[4] += 1
	medium_red[4] = medium_red[4]/17

	for i in range(26,43):
		medium_red[5] += reduction[i][reduction[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[5] += 1
	medium_red[5] = medium_red[5]/17



	for i in range(26,43):
		medium_red[6] += reduction[i][reduction[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[6] += 1
	medium_red[6] = medium_red[6]/17

	for i in range(26,43):
		medium_red[7] += reduction[i][reduction[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[7] += 1
	medium_red[7] = medium_red[7]/17

	for i in range(26,43):
		medium_red[8] += reduction[i][reduction[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[8] += 1
	medium_red[8] = medium_red[8]/17

	for i in range(26,43):
		medium_red[9] += reduction[i][reduction[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[9] += 1
	medium_red[9] = medium_red[9]/17

	for i in range(26,43):
		medium_red[10] += reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[10] += 1
	medium_red[10] = medium_red[10]/17

	for i in range(26,43):
		medium_red[11] += reduction[i][reduction[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[11] += 1
	medium_red[11] = medium_red[11]/17

	

	for i in range(26,43):
		medium_red[12] += reduction[i][reduction[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[12] += 1
	medium_red[12] = medium_red[12]/17

	for i in range(26,43):
		medium_red[13] += reduction[i][reduction[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[13] += 1
	medium_red[13] = medium_red[13]/17

	for i in range(26,43):
		medium_red[14] += reduction[i][reduction[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[14] += 1
	medium_red[14] = medium_red[14]/17

	for i in range(26,43):
		medium_red[15] += reduction[i][reduction[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[15] += 1
	medium_red[15] = medium_red[15]/17

	for i in range(26,43):
		medium_red[16] += reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[16] += 1
	medium_red[16] = medium_red[16]/17

	for i in range(26,43):
		medium_red[17] += reduction[i][reduction[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[17] += 1
	medium_red[17] = medium_red[17]/17



	for i in range(26,43):
		medium_red[18] += reduction[i][reduction[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[18] += 1
	medium_red[18] = medium_red[18]/17

	for i in range(26,43):
		medium_red[19] += reduction[i][reduction[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[19] += 1
	medium_red[19] = medium_red[19]/17

	for i in range(26,43):
		medium_red[20] += reduction[i][reduction[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[20] += 1
	medium_red[20] = medium_red[20]/17

	for i in range(26,43):
		medium_red[21] += reduction[i][reduction[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[21] += 1
	medium_red[21] = medium_red[21]/17

	for i in range(26,43):
		medium_red[22] += reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[22] += 1
	medium_red[22] = medium_red[22]/17

	for i in range(26,43):
		medium_red[23] += reduction[i][reduction[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_red[23] += 1
	medium_red[23] = medium_red[23]/17








	for i in range(43,45):
		large_red[0] += reduction[i][reduction[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[0] += 1
	large_red[0] = large_red[0]/2

	for i in range(43,45):
		large_red[1] += reduction[i][reduction[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[1] += 1
	large_red[1] = large_red[1]/2

	for i in range(43,45):
		large_red[2] += reduction[i][reduction[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[2] += 1
	large_red[2] = large_red[2]/2

	for i in range(43,45):
		large_red[3] += reduction[i][reduction[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[3] += 1
	large_red[3] = large_red[3]/2

	for i in range(43,45):
		large_red[4] += reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[4] += 1
	large_red[4] = large_red[4]/2

	for i in range(43,45):
		large_red[5] += reduction[i][reduction[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[5] += 1
	large_red[5] = large_red[5]/2



	for i in range(43,45):
		large_red[6] += reduction[i][reduction[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[6] += 1
	large_red[6] = large_red[6]/2

	for i in range(43,45):
		large_red[7] += reduction[i][reduction[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[7] += 1
	large_red[7] = large_red[7]/2

	for i in range(43,45):
		large_red[8] += reduction[i][reduction[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[8] += 1
	large_red[8] = large_red[8]/2

	for i in range(43,45):
		large_red[9] += reduction[i][reduction[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[9] += 1
	large_red[9] = large_red[9]/2

	for i in range(43,45):
		large_red[10] += reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[10] += 1
	large_red[10] = large_red[10]/2

	for i in range(43,45):
		large_red[11] += reduction[i][reduction[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[11] += 1
	large_red[11] = large_red[11]/2

	

	for i in range(43,45):
		large_red[12] += reduction[i][reduction[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[12] += 1
	large_red[12] = large_red[12]/2

	for i in range(43,45):
		large_red[13] += reduction[i][reduction[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[13] += 1
	large_red[13] = large_red[13]/2

	for i in range(43,45):
		large_red[14] += reduction[i][reduction[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[14] += 1
	large_red[14] = large_red[14]/2

	for i in range(43,45):
		large_red[15] += reduction[i][reduction[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[15] += 1
	large_red[15] = large_red[15]/2

	for i in range(43,45):
		large_red[16] += reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[16] += 1
	large_red[16] = large_red[16]/2

	for i in range(43,45):
		large_red[17] += reduction[i][reduction[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[17] += 1
	large_red[17] = large_red[17]/2



	for i in range(43,45):
		large_red[18] += reduction[i][reduction[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[18] += 1
	large_red[18] = large_red[18]/2

	for i in range(43,45):
		large_red[19] += reduction[i][reduction[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[19] += 1
	large_red[19] = large_red[19]/2

	for i in range(43,45):
		large_red[20] += reduction[i][reduction[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[20] += 1
	large_red[20] = large_red[20]/2

	for i in range(43,45):
		large_red[21] += reduction[i][reduction[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[21] += 1
	large_red[21] = large_red[21]/2

	for i in range(43,45):
		large_red[22] += reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[22] += 1
	large_red[22] = large_red[22]/2

	for i in range(43,45):
		large_red[23] += reduction[i][reduction[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((reduction[i][reduction[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_red[23] += 1
	large_red[23] = large_red[23]/2


	print("reduction")
	print("*"*100)
	print(small_red)
	print(medium_red)
	print(large_red)


	print(first_small_red)
	print(first_medium_red)
	print(first_large_red)

	print("*"*100)

	small_accred = [0] * 24
	first_small_accred = [0] * 24
	medium_accred = [0] * 24
	first_medium_accred = [0] * 24
	large_accred = [0] * 24
	first_large_accred = [0] * 24



	for i in range(0,26):
		small_accred[0] += acc_red[i][acc_red[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[0] += 1
	small_accred[0] = small_accred[0]/26

	for i in range(0,26):
		small_accred[1] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[1] += 1
	small_accred[1] = small_accred[1]/26

	for i in range(0,26):
		small_accred[2] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[2] += 1
	small_accred[2] = small_accred[2]/26

	for i in range(0,26):
		small_accred[3] += acc_red[i][acc_red[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[3] += 1
	small_accred[3] = small_accred[3]/26

	for i in range(0,26):
		small_accred[4] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[4] += 1
	small_accred[4] = small_accred[4]/26

	for i in range(0,26):
		small_accred[5] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[5] += 1
	small_accred[5] = small_accred[5]/26



	for i in range(0,26):
		small_accred[6] += acc_red[i][acc_red[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[6] += 1
	small_accred[6] = small_accred[6]/26

	for i in range(0,26):
		small_accred[7] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[7] += 1
	small_accred[7] = small_accred[7]/26

	for i in range(0,26):
		small_accred[8] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[8] += 1
	small_accred[8] = small_accred[8]/26

	for i in range(0,26):
		small_accred[9] += acc_red[i][acc_red[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[9] += 1
	small_accred[9] = small_accred[9]/26

	for i in range(0,26):
		small_accred[10] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[10] += 1
	small_accred[10] = small_accred[10]/26

	for i in range(0,26):
		small_accred[11] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[11] += 1
	small_accred[11] = small_accred[11]/26

	

	for i in range(0,26):
		small_accred[12] += acc_red[i][acc_red[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[12] += 1
	small_accred[12] = small_accred[12]/26

	for i in range(0,26):
		small_accred[13] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[13] += 1
	small_accred[13] = small_accred[13]/26

	for i in range(0,26):
		small_accred[14] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[14] += 1
	small_accred[14] = small_accred[14]/26

	for i in range(0,26):
		small_accred[15] += acc_red[i][acc_red[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[15] += 1
	small_accred[15] = small_accred[15]/26

	for i in range(0,26):
		small_accred[16] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[16] += 1
	small_accred[16] = small_accred[16]/26

	for i in range(0,26):
		small_accred[17] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[17] += 1
	small_accred[17] = small_accred[17]/26



	for i in range(0,26):
		small_accred[18] += acc_red[i][acc_red[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[18] += 1
	small_accred[18] = small_accred[18]/26

	for i in range(0,26):
		small_accred[19] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[19] += 1
	small_accred[19] = small_accred[19]/26

	for i in range(0,26):
		small_accred[20] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[20] += 1
	small_accred[20] = small_accred[20]/26

	for i in range(0,26):
		small_accred[21] += acc_red[i][acc_red[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[21] += 1
	small_accred[21] = small_accred[21]/26

	for i in range(0,26):
		small_accred[22] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[22] += 1
	small_accred[22] = small_accred[22]/26

	for i in range(0,26):
		small_accred[23] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_accred[23] += 1
	small_accred[23] = small_accred[23]/26





	for i in range(26,43):
		medium_accred[0] += acc_red[i][acc_red[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[0] += 1
	medium_accred[0] = medium_accred[0]/17

	for i in range(26,43):
		medium_accred[1] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[1] += 1
	medium_accred[1] = medium_accred[1]/17

	for i in range(26,43):
		medium_accred[2] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[2] += 1
	medium_accred[2] = medium_accred[2]/17

	for i in range(26,43):
		medium_accred[3] += acc_red[i][acc_red[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[3] += 1
	medium_accred[3] = medium_accred[3]/17

	for i in range(26,43):
		medium_accred[4] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[4] += 1
	medium_accred[4] = medium_accred[4]/17

	for i in range(26,43):
		medium_accred[5] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[5] += 1
	medium_accred[5] = medium_accred[5]/17



	for i in range(26,43):
		medium_accred[6] += acc_red[i][acc_red[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[6] += 1
	medium_accred[6] = medium_accred[6]/17

	for i in range(26,43):
		medium_accred[7] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[7] += 1
	medium_accred[7] = medium_accred[7]/17

	for i in range(26,43):
		medium_accred[8] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[8] += 1
	medium_accred[8] = medium_accred[8]/17

	for i in range(26,43):
		medium_accred[9] += acc_red[i][acc_red[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[9] += 1
	medium_accred[9] = medium_accred[9]/17

	for i in range(26,43):
		medium_accred[10] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[10] += 1
	medium_accred[10] = medium_accred[10]/17

	for i in range(26,43):
		medium_accred[11] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[11] += 1
	medium_accred[11] = medium_accred[11]/17

	

	for i in range(26,43):
		medium_accred[12] += acc_red[i][acc_red[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[12] += 1
	medium_accred[12] = medium_accred[12]/17

	for i in range(26,43):
		medium_accred[13] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[13] += 1
	medium_accred[13] = medium_accred[13]/17

	for i in range(26,43):
		medium_accred[14] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[14] += 1
	medium_accred[14] = medium_accred[14]/17

	for i in range(26,43):
		medium_accred[15] += acc_red[i][acc_red[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[15] += 1
	medium_accred[15] = medium_accred[15]/17

	for i in range(26,43):
		medium_accred[16] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[16] += 1
	medium_accred[16] = medium_accred[16]/17

	for i in range(26,43):
		medium_accred[17] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[17] += 1
	medium_accred[17] = medium_accred[17]/17



	for i in range(26,43):
		medium_accred[18] += acc_red[i][acc_red[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[18] += 1
	medium_accred[18] = medium_accred[18]/17

	for i in range(26,43):
		medium_accred[19] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[19] += 1
	medium_accred[19] = medium_accred[19]/17

	for i in range(26,43):
		medium_accred[20] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[20] += 1
	medium_accred[20] = medium_accred[20]/17

	for i in range(26,43):
		medium_accred[21] += acc_red[i][acc_red[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[21] += 1
	medium_accred[21] = medium_accred[21]/17

	for i in range(26,43):
		medium_accred[22] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[22] += 1
	medium_accred[22] = medium_accred[22]/17

	for i in range(26,43):
		medium_accred[23] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_accred[23] += 1
	medium_accred[23] = medium_accred[23]/17








	for i in range(43,45):
		large_accred[0] += acc_red[i][acc_red[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[0] += 1
	large_accred[0] = large_accred[0]/2

	for i in range(43,45):
		large_accred[1] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[1] += 1
	large_accred[1] = large_accred[1]/2

	for i in range(43,45):
		large_accred[2] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[2] += 1
	large_accred[2] = large_accred[2]/2

	for i in range(43,45):
		large_accred[3] += acc_red[i][acc_red[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[3] += 1
	large_accred[3] = large_accred[3]/2

	for i in range(43,45):
		large_accred[4] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[4] += 1
	large_accred[4] = large_accred[4]/2

	for i in range(43,45):
		large_accred[5] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[5] += 1
	large_accred[5] = large_accred[5]/2



	for i in range(43,45):
		large_accred[6] += acc_red[i][acc_red[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[6] += 1
	large_accred[6] = large_accred[6]/2

	for i in range(43,45):
		large_accred[7] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[7] += 1
	large_accred[7] = large_accred[7]/2

	for i in range(43,45):
		large_accred[8] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[8] += 1
	large_accred[8] = large_accred[8]/2

	for i in range(43,45):
		large_accred[9] += acc_red[i][acc_red[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[9] += 1
	large_accred[9] = large_accred[9]/2

	for i in range(43,45):
		large_accred[10] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[10] += 1
	large_accred[10] = large_accred[10]/2

	for i in range(43,45):
		large_accred[11] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[11] += 1
	large_accred[11] = large_accred[11]/2

	

	for i in range(43,45):
		large_accred[12] += acc_red[i][acc_red[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[12] += 1
	large_accred[12] = large_accred[12]/2

	for i in range(43,45):
		large_accred[13] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[13] += 1
	large_accred[13] = large_accred[13]/2

	for i in range(43,45):
		large_accred[14] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[14] += 1
	large_accred[14] = large_accred[14]/2

	for i in range(43,45):
		large_accred[15] += acc_red[i][acc_red[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[15] += 1
	large_accred[15] = large_accred[15]/2

	for i in range(43,45):
		large_accred[16] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[16] += 1
	large_accred[16] = large_accred[16]/2

	for i in range(43,45):
		large_accred[17] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[17] += 1
	large_accred[17] = large_accred[17]/2



	for i in range(43,45):
		large_accred[18] += acc_red[i][acc_red[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[18] += 1
	large_accred[18] = large_accred[18]/2

	for i in range(43,45):
		large_accred[19] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[19] += 1
	large_accred[19] = large_accred[19]/2

	for i in range(43,45):
		large_accred[20] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[20] += 1
	large_accred[20] = large_accred[20]/2

	for i in range(43,45):
		large_accred[21] += acc_red[i][acc_red[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[21] += 1
	large_accred[21] = large_accred[21]/2

	for i in range(43,45):
		large_accred[22] += acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[22] += 1
	large_accred[22] = large_accred[22]/2

	for i in range(43,45):
		large_accred[23] += acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((acc_red[i][acc_red[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_accred[23] += 1
	large_accred[23] = large_accred[23]/2


	print("acc_red")
	print("*"*100)
	print(small_accred)
	print(medium_accred)
	print(large_accred)


	print(first_small_accred)
	print(first_medium_accred)
	print(first_large_accred)

	print("*"*100)

	small_kapred = [0] * 24
	first_small_kapred = [0] * 24
	medium_kapred = [0] * 24
	first_medium_kapred = [0] * 24
	large_kapred = [0] * 24
	first_large_kapred = [0] * 24



	for i in range(0,26):
		small_kapred[0] += kap_red[i][kap_red[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[0] += 1
	small_kapred[0] = small_kapred[0]/26

	for i in range(0,26):
		small_kapred[1] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[1] += 1
	small_kapred[1] = small_kapred[1]/26

	for i in range(0,26):
		small_kapred[2] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[2] += 1
	small_kapred[2] = small_kapred[2]/26

	for i in range(0,26):
		small_kapred[3] += kap_red[i][kap_red[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[3] += 1
	small_kapred[3] = small_kapred[3]/26

	for i in range(0,26):
		small_kapred[4] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[4] += 1
	small_kapred[4] = small_kapred[4]/26

	for i in range(0,26):
		small_kapred[5] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[5] += 1
	small_kapred[5] = small_kapred[5]/26



	for i in range(0,26):
		small_kapred[6] += kap_red[i][kap_red[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[6] += 1
	small_kapred[6] = small_kapred[6]/26

	for i in range(0,26):
		small_kapred[7] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[7] += 1
	small_kapred[7] = small_kapred[7]/26

	for i in range(0,26):
		small_kapred[8] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[8] += 1
	small_kapred[8] = small_kapred[8]/26

	for i in range(0,26):
		small_kapred[9] += kap_red[i][kap_red[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[9] += 1
	small_kapred[9] = small_kapred[9]/26

	for i in range(0,26):
		small_kapred[10] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[10] += 1
	small_kapred[10] = small_kapred[10]/26

	for i in range(0,26):
		small_kapred[11] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[11] += 1
	small_kapred[11] = small_kapred[11]/26

	

	for i in range(0,26):
		small_kapred[12] += kap_red[i][kap_red[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[12] += 1
	small_kapred[12] = small_kapred[12]/26

	for i in range(0,26):
		small_kapred[13] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[13] += 1
	small_kapred[13] = small_kapred[13]/26

	for i in range(0,26):
		small_kapred[14] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[14] += 1
	small_kapred[14] = small_kapred[14]/26

	for i in range(0,26):
		small_kapred[15] += kap_red[i][kap_red[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[15] += 1
	small_kapred[15] = small_kapred[15]/26

	for i in range(0,26):
		small_kapred[16] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[16] += 1
	small_kapred[16] = small_kapred[16]/26

	for i in range(0,26):
		small_kapred[17] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[17] += 1
	small_kapred[17] = small_kapred[17]/26



	for i in range(0,26):
		small_kapred[18] += kap_red[i][kap_red[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[18] += 1
	small_kapred[18] = small_kapred[18]/26

	for i in range(0,26):
		small_kapred[19] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[19] += 1
	small_kapred[19] = small_kapred[19]/26

	for i in range(0,26):
		small_kapred[20] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[20] += 1
	small_kapred[20] = small_kapred[20]/26

	for i in range(0,26):
		small_kapred[21] += kap_red[i][kap_red[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[21] += 1
	small_kapred[21] = small_kapred[21]/26

	for i in range(0,26):
		small_kapred[22] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[22] += 1
	small_kapred[22] = small_kapred[22]/26

	for i in range(0,26):
		small_kapred[23] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_kapred[23] += 1
	small_kapred[23] = small_kapred[23]/26





	for i in range(26,43):
		medium_kapred[0] += kap_red[i][kap_red[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[0] += 1
	medium_kapred[0] = medium_kapred[0]/17

	for i in range(26,43):
		medium_kapred[1] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[1] += 1
	medium_kapred[1] = medium_kapred[1]/17

	for i in range(26,43):
		medium_kapred[2] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[2] += 1
	medium_kapred[2] = medium_kapred[2]/17

	for i in range(26,43):
		medium_kapred[3] += kap_red[i][kap_red[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[3] += 1
	medium_kapred[3] = medium_kapred[3]/17

	for i in range(26,43):
		medium_kapred[4] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[4] += 1
	medium_kapred[4] = medium_kapred[4]/17

	for i in range(26,43):
		medium_kapred[5] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[5] += 1
	medium_kapred[5] = medium_kapred[5]/17



	for i in range(26,43):
		medium_kapred[6] += kap_red[i][kap_red[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[6] += 1
	medium_kapred[6] = medium_kapred[6]/17

	for i in range(26,43):
		medium_kapred[7] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[7] += 1
	medium_kapred[7] = medium_kapred[7]/17

	for i in range(26,43):
		medium_kapred[8] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[8] += 1
	medium_kapred[8] = medium_kapred[8]/17

	for i in range(26,43):
		medium_kapred[9] += kap_red[i][kap_red[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[9] += 1
	medium_kapred[9] = medium_kapred[9]/17

	for i in range(26,43):
		medium_kapred[10] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[10] += 1
	medium_kapred[10] = medium_kapred[10]/17

	for i in range(26,43):
		medium_kapred[11] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[11] += 1
	medium_kapred[11] = medium_kapred[11]/17

	

	for i in range(26,43):
		medium_kapred[12] += kap_red[i][kap_red[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[12] += 1
	medium_kapred[12] = medium_kapred[12]/17

	for i in range(26,43):
		medium_kapred[13] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[13] += 1
	medium_kapred[13] = medium_kapred[13]/17

	for i in range(26,43):
		medium_kapred[14] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[14] += 1
	medium_kapred[14] = medium_kapred[14]/17

	for i in range(26,43):
		medium_kapred[15] += kap_red[i][kap_red[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[15] += 1
	medium_kapred[15] = medium_kapred[15]/17

	for i in range(26,43):
		medium_kapred[16] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[16] += 1
	medium_kapred[16] = medium_kapred[16]/17

	for i in range(26,43):
		medium_kapred[17] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[17] += 1
	medium_kapred[17] = medium_kapred[17]/17



	for i in range(26,43):
		medium_kapred[18] += kap_red[i][kap_red[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[18] += 1
	medium_kapred[18] = medium_kapred[18]/17

	for i in range(26,43):
		medium_kapred[19] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[19] += 1
	medium_kapred[19] = medium_kapred[19]/17

	for i in range(26,43):
		medium_kapred[20] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[20] += 1
	medium_kapred[20] = medium_kapred[20]/17

	for i in range(26,43):
		medium_kapred[21] += kap_red[i][kap_red[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[21] += 1
	medium_kapred[21] = medium_kapred[21]/17

	for i in range(26,43):
		medium_kapred[22] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[22] += 1
	medium_kapred[22] = medium_kapred[22]/17

	for i in range(26,43):
		medium_kapred[23] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_kapred[23] += 1
	medium_kapred[23] = medium_kapred[23]/17








	for i in range(43,45):
		large_kapred[0] += kap_red[i][kap_red[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[0] += 1
	large_kapred[0] = large_kapred[0]/2

	for i in range(43,45):
		large_kapred[1] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[1] += 1
	large_kapred[1] = large_kapred[1]/2

	for i in range(43,45):
		large_kapred[2] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[2] += 1
	large_kapred[2] = large_kapred[2]/2

	for i in range(43,45):
		large_kapred[3] += kap_red[i][kap_red[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[3] += 1
	large_kapred[3] = large_kapred[3]/2

	for i in range(43,45):
		large_kapred[4] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[4] += 1
	large_kapred[4] = large_kapred[4]/2

	for i in range(43,45):
		large_kapred[5] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[5] += 1
	large_kapred[5] = large_kapred[5]/2



	for i in range(43,45):
		large_kapred[6] += kap_red[i][kap_red[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[6] += 1
	large_kapred[6] = large_kapred[6]/2

	for i in range(43,45):
		large_kapred[7] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[7] += 1
	large_kapred[7] = large_kapred[7]/2

	for i in range(43,45):
		large_kapred[8] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[8] += 1
	large_kapred[8] = large_kapred[8]/2

	for i in range(43,45):
		large_kapred[9] += kap_red[i][kap_red[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[9] += 1
	large_kapred[9] = large_kapred[9]/2

	for i in range(43,45):
		large_kapred[10] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[10] += 1
	large_kapred[10] = large_kapred[10]/2

	for i in range(43,45):
		large_kapred[11] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[11] += 1
	large_kapred[11] = large_kapred[11]/2

	

	for i in range(43,45):
		large_kapred[12] += kap_red[i][kap_red[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[12] += 1
	large_kapred[12] = large_kapred[12]/2

	for i in range(43,45):
		large_kapred[13] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[13] += 1
	large_kapred[13] = large_kapred[13]/2

	for i in range(43,45):
		large_kapred[14] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[14] += 1
	large_kapred[14] = large_kapred[14]/2

	for i in range(43,45):
		large_kapred[15] += kap_red[i][kap_red[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[15] += 1
	large_kapred[15] = large_kapred[15]/2

	for i in range(43,45):
		large_kapred[16] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[16] += 1
	large_kapred[16] = large_kapred[16]/2

	for i in range(43,45):
		large_kapred[17] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[17] += 1
	large_kapred[17] = large_kapred[17]/2



	for i in range(43,45):
		large_kapred[18] += kap_red[i][kap_red[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[18] += 1
	large_kapred[18] = large_kapred[18]/2

	for i in range(43,45):
		large_kapred[19] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[19] += 1
	large_kapred[19] = large_kapred[19]/2

	for i in range(43,45):
		large_kapred[20] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[20] += 1
	large_kapred[20] = large_kapred[20]/2

	for i in range(43,45):
		large_kapred[21] += kap_red[i][kap_red[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[21] += 1
	large_kapred[21] = large_kapred[21]/2

	for i in range(43,45):
		large_kapred[22] += kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[22] += 1
	large_kapred[22] = large_kapred[22]/2

	for i in range(43,45):
		large_kapred[23] += kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((kap_red[i][kap_red[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_kapred[23] += 1
	large_kapred[23] = large_kapred[23]/2


	print("kap_red")
	print("*"*100)
	print(small_kapred)
	print(medium_kapred)
	print(large_kapred)


	print(first_small_kapred)
	print(first_medium_kapred)
	print(first_large_kapred)

	print("*"*100)

	small_time = [0] * 24
	first_small_time = [0] * 24
	medium_time = [0] * 24
	first_medium_time = [0] * 24
	large_time = [0] * 24
	first_large_time = [0] * 24



	for i in range(0,26):
		small_time[0] += time[i][time[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[0] += 1
	small_time[0] = small_time[0]/26

	for i in range(0,26):
		small_time[1] += time[i][time[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[1] += 1
	small_time[1] = small_time[1]/26

	for i in range(0,26):
		small_time[2] += time[i][time[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[2] += 1
	small_time[2] = small_time[2]/26

	for i in range(0,26):
		small_time[3] += time[i][time[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[3] += 1
	small_time[3] = small_time[3]/26

	for i in range(0,26):
		small_time[4] += time[i][time[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[4] += 1
	small_time[4] = small_time[4]/26

	for i in range(0,26):
		small_time[5] += time[i][time[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[5] += 1
	small_time[5] = small_time[5]/26



	for i in range(0,26):
		small_time[6] += time[i][time[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[6] += 1
	small_time[6] = small_time[6]/26

	for i in range(0,26):
		small_time[7] += time[i][time[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[7] += 1
	small_time[7] = small_time[7]/26

	for i in range(0,26):
		small_time[8] += time[i][time[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[8] += 1
	small_time[8] = small_time[8]/26

	for i in range(0,26):
		small_time[9] += time[i][time[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[9] += 1
	small_time[9] = small_time[9]/26

	for i in range(0,26):
		small_time[10] += time[i][time[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[10] += 1
	small_time[10] = small_time[10]/26

	for i in range(0,26):
		small_time[11] += time[i][time[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[11] += 1
	small_time[11] = small_time[11]/26

	

	for i in range(0,26):
		small_time[12] += time[i][time[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[12] += 1
	small_time[12] = small_time[12]/26

	for i in range(0,26):
		small_time[13] += time[i][time[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[13] += 1
	small_time[13] = small_time[13]/26

	for i in range(0,26):
		small_time[14] += time[i][time[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[14] += 1
	small_time[14] = small_time[14]/26

	for i in range(0,26):
		small_time[15] += time[i][time[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[15] += 1
	small_time[15] = small_time[15]/26

	for i in range(0,26):
		small_time[16] += time[i][time[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[16] += 1
	small_time[16] = small_time[16]/26

	for i in range(0,26):
		small_time[17] += time[i][time[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[17] += 1
	small_time[17] = small_time[17]/26



	for i in range(0,26):
		small_time[18] += time[i][time[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[18] += 1
	small_time[18] = small_time[18]/26

	for i in range(0,26):
		small_time[19] += time[i][time[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[19] += 1
	small_time[19] = small_time[19]/26

	for i in range(0,26):
		small_time[20] += time[i][time[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[20] += 1
	small_time[20] = small_time[20]/26

	for i in range(0,26):
		small_time[21] += time[i][time[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[21] += 1
	small_time[21] = small_time[21]/26

	for i in range(0,26):
		small_time[22] += time[i][time[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[22] += 1
	small_time[22] = small_time[22]/26

	for i in range(0,26):
		small_time[23] += time[i][time[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_small_time[23] += 1
	small_time[23] = small_time[23]/26





	for i in range(26,43):
		medium_time[0] += time[i][time[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[0] += 1
	medium_time[0] = medium_time[0]/17

	for i in range(26,43):
		medium_time[1] += time[i][time[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[1] += 1
	medium_time[1] = medium_time[1]/17

	for i in range(26,43):
		medium_time[2] += time[i][time[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[2] += 1
	medium_time[2] = medium_time[2]/17

	for i in range(26,43):
		medium_time[3] += time[i][time[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[3] += 1
	medium_time[3] = medium_time[3]/17

	for i in range(26,43):
		medium_time[4] += time[i][time[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[4] += 1
	medium_time[4] = medium_time[4]/17

	for i in range(26,43):
		medium_time[5] += time[i][time[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[5] += 1
	medium_time[5] = medium_time[5]/17



	for i in range(26,43):
		medium_time[6] += time[i][time[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[6] += 1
	medium_time[6] = medium_time[6]/17

	for i in range(26,43):
		medium_time[7] += time[i][time[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[7] += 1
	medium_time[7] = medium_time[7]/17

	for i in range(26,43):
		medium_time[8] += time[i][time[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[8] += 1
	medium_time[8] = medium_time[8]/17

	for i in range(26,43):
		medium_time[9] += time[i][time[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[9] += 1
	medium_time[9] = medium_time[9]/17

	for i in range(26,43):
		medium_time[10] += time[i][time[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[10] += 1
	medium_time[10] = medium_time[10]/17

	for i in range(26,43):
		medium_time[11] += time[i][time[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[11] += 1
	medium_time[11] = medium_time[11]/17

	

	for i in range(26,43):
		medium_time[12] += time[i][time[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[12] += 1
	medium_time[12] = medium_time[12]/17

	for i in range(26,43):
		medium_time[13] += time[i][time[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[13] += 1
	medium_time[13] = medium_time[13]/17

	for i in range(26,43):
		medium_time[14] += time[i][time[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[14] += 1
	medium_time[14] = medium_time[14]/17

	for i in range(26,43):
		medium_time[15] += time[i][time[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[15] += 1
	medium_time[15] = medium_time[15]/17

	for i in range(26,43):
		medium_time[16] += time[i][time[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[16] += 1
	medium_time[16] = medium_time[16]/17

	for i in range(26,43):
		medium_time[17] += time[i][time[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[17] += 1
	medium_time[17] = medium_time[17]/17



	for i in range(26,43):
		medium_time[18] += time[i][time[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[18] += 1
	medium_time[18] = medium_time[18]/17

	for i in range(26,43):
		medium_time[19] += time[i][time[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[19] += 1
	medium_time[19] = medium_time[19]/17

	for i in range(26,43):
		medium_time[20] += time[i][time[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[20] += 1
	medium_time[20] = medium_time[20]/17

	for i in range(26,43):
		medium_time[21] += time[i][time[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[21] += 1
	medium_time[21] = medium_time[21]/17

	for i in range(26,43):
		medium_time[22] += time[i][time[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[22] += 1
	medium_time[22] = medium_time[22]/17

	for i in range(26,43):
		medium_time[23] += time[i][time[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_medium_time[23] += 1
	medium_time[23] = medium_time[23]/17








	for i in range(43,45):
		large_time[0] += time[i][time[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[0] += 1
	large_time[0] = large_time[0]/2

	for i in range(43,45):
		large_time[1] += time[i][time[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[1] += 1
	large_time[1] = large_time[1]/2

	for i in range(43,45):
		large_time[2] += time[i][time[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[2] += 1
	large_time[2] = large_time[2]/2

	for i in range(43,45):
		large_time[3] += time[i][time[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[3] += 1
	large_time[3] = large_time[3]/2

	for i in range(43,45):
		large_time[4] += time[i][time[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[4] += 1
	large_time[4] = large_time[4]/2

	for i in range(43,45):
		large_time[5] += time[i][time[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-rss-gga"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[5] += 1
	large_time[5] = large_time[5]/2



	for i in range(43,45):
		large_time[6] += time[i][time[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[6] += 1
	large_time[6] = large_time[6]/2

	for i in range(43,45):
		large_time[7] += time[i][time[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[7] += 1
	large_time[7] = large_time[7]/2

	for i in range(43,45):
		large_time[8] += time[i][time[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[8] += 1
	large_time[8] = large_time[8]/2

	for i in range(43,45):
		large_time[9] += time[i][time[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[9] += 1
	large_time[9] = large_time[9]/2

	for i in range(43,45):
		large_time[10] += time[i][time[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[10] += 1
	large_time[10] = large_time[10]/2

	for i in range(43,45):
		large_time[11] += time[i][time[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-rss-ssga"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[11] += 1
	large_time[11] = large_time[11]/2

	

	for i in range(43,45):
		large_time[12] += time[i][time[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[12] += 1
	large_time[12] = large_time[12]/2

	for i in range(43,45):
		large_time[13] += time[i][time[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[13] += 1
	large_time[13] = large_time[13]/2

	for i in range(43,45):
		large_time[14] += time[i][time[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[14] += 1
	large_time[14] = large_time[14]/2

	for i in range(43,45):
		large_time[15] += time[i][time[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[15] += 1
	large_time[15] = large_time[15]/2

	for i in range(43,45):
		large_time[16] += time[i][time[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[16] += 1
	large_time[16] = large_time[16]/2

	for i in range(43,45):
		large_time[17] += time[i][time[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-rss-ma"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[17] += 1
	large_time[17] = large_time[17]/2



	for i in range(43,45):
		large_time[18] += time[i][time[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[18] += 1
	large_time[18] = large_time[18]/2

	for i in range(43,45):
		large_time[19] += time[i][time[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[19] += 1
	large_time[19] = large_time[19]/2

	for i in range(43,45):
		large_time[20] += time[i][time[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[20] += 1
	large_time[20] = large_time[20]/2

	for i in range(43,45):
		large_time[21] += time[i][time[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[21] += 1
	large_time[21] = large_time[21]/2

	for i in range(43,45):
		large_time[22] += time[i][time[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Cnn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[22] += 1
	large_time[22] = large_time[22]/2

	for i in range(43,45):
		large_time[23] += time[i][time[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1
		if ((time[i][time[i].iloc[:,9] == "Enn-rss-chc"].dropna(how='all').index[0] + 1) == 1):
			first_large_time[23] += 1
	large_time[23] = large_time[23]/2


	print("time")
	print("*"*100)
	print(small_time)
	print(medium_time)
	print(large_time)


	print(first_small_time)
	print(first_medium_time)
	print(first_large_time)

	print("*"*100)