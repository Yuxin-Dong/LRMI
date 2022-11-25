import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from pyitlib import discrete_random_variable as drv
import math

data = np.loadtxt('dataset/madelon.txt', dtype=np.int32)
features = [
	[241, 338, 378, 318, 153, 455, 48, 493, 28, 298, ], # MRMI
	[241, 338, 378, 318, 153, 455, 48, 493, 28, 124], # LRMI
	[105, 90, 423, 276, 404, 228, 332, 168, 173, 402, ], # MIFS
	[105, 493, 462, 175, 281, 382, 136, 107, 467, 478, ], # FOU
	[105, 90, 423, 276, 404, 228, 332, 168, 173, 283, ], # MRMR
	[105, 453, 64, 442, 153, 475, 338, 455, 378, 493, ], # JMI
	[105, 453, 433, 411, 336, 442, 475, 455, 75, 13, ], # CMIM
	[105, 433, 493, 336, 338, 442, 475, 455, 378, 453, ], # DISR
]
# data = np.loadtxt('dataset/semeion/semeion.txt', dtype=np.int32)
# features = [
# 	[161, 78, 81, 228, 76, 5, 190, 118, 234, 14, ], # MRMI
# 	[161, 78, 81, 228, 76, 5, 190, 15, 118, 234, ], # LRMI
# 	[161, 78, 81, 228, 76, 7, 190, 129, 193, 134, ], # MIFS
# 	[161, 78, 81, 228, 100, 5, 113, 115, 85, 70, ], # FOU
# 	[161, 78, 81, 177, 145, 111, 228, 62, 144, 193, ], # MRMR
# 	[161, 78, 81, 177, 145, 62, 144, 193, 110, 129, ], # JMI
# 	[161, 78, 81, 228, 7, 76, 126, 129, 190, 193, ], # CMIM
# 	[161, 78, 145, 177, 111, 81, 62, 144, 193, 94, ], # DISR
# ]
# data = np.loadtxt('dataset/statlog/statlog.txt', dtype=np.int32)
# features = [
# 	[17, 16, 19, 27, 11, 32, 2, 0, 34, 9, ], # MRMI
# 	[17, 16, 19, 11, 27, 35, 2, 32, 0, 9, ], # LRMI
# 	[17, 20, 24, 35, 0, 9, 26, 32, 2, 8, ], # MIFS
# 	[17, 20, 34, 10, 0, 26, 2, 24, 30, 9, ], # FOU
# 	[17, 24, 8, 35, 1, 20, 15, 32, 12, 9, ], # MRMR
# 	[17, 20, 19, 12, 5, 35, 16, 33, 15, 13, ], # JMI
# 	[17, 20, 19, 26, 2, 18, 10, 34, 3, 0, ], # CMIM
# 	[16, 17, 19, 20, 21, 12, 23, 13, 28, 15, ], # DISR
# ]
# data = np.loadtxt('dataset/optdigits/optdigits.txt', dtype=np.int32)
# features = [
# 	[42, 21, 43, 26, 10, 61, 27, 19, 37, 5, ], # MRMI
# 	[42, 21, 43, 26, 10, 61, 27, 52, 36, 5, ], # LRMI
# 	[42, 21, 61, 38, 26, 10, 43, 27, 0, 39, ], # MIFS
# 	[42, 21, 27, 44, 37, 45, 29, 52, 53, 51, ], # FOU
# 	[42, 21, 61, 30, 26, 43, 28, 10, 34, 38, ], # MRMR
# 	[42, 21, 43, 28, 61, 26, 34, 36, 20, 29, ], # JMI
# 	[42, 21, 43, 61, 20, 26, 36, 2, 27, 13, ], # CMIM
# 	[42, 30, 43, 21, 28, 34, 36, 20, 38, 54, ], # DISR
# ]

X = data[:, :-1]
y = data[:, -1]

# scaler = StandardScaler()
# scaler.fit(X)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

feat_ids = []
feat_map = {}

for m in features:
	feat_id = []
	for i in range(len(m)):
		feat = tuple(set(m[:i+1]))
		if feat in feat_map:
			fid = feat_map[feat]
		else:
			fid = len(feat_map)
			feat_map[feat] = fid
		feat_id.append(fid)
	feat_ids.append(feat_id)

feat_res = np.zeros(len(feat_map))

classifier = KNeighborsClassifier(n_neighbors=3)

for feat, fid in feat_map.items():
	error = []
	for index_train, index_test in KFold(n_splits=10, shuffle=True).split(X):
		X_train, X_test = X[index_train, :][:, feat], X[index_test, :][:, feat]
		y_train, y_test = y[index_train], y[index_test]
		if i == 0:
			X_train = X_train.reshape(-1, 1)
			X_test = X_test.reshape(-1, 1)

		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)

		# print(confusion_matrix(y_test, y_pred))
		# print(classification_report(y_test, y_pred))
		error.append(np.mean(y_pred != y_test))

	error = np.array(error) * 100
	m = np.mean(error)
	sd = np.sqrt(np.square(error - m).mean())
	# print(m)
	print(feat, '%.2f (%.2f)' % (m, sd))
	feat_res[fid] = m

for m in feat_ids:
	for fid in m:
		print('%.2f' % feat_res[fid], end='\t')
	print()

ranking = np.zeros((len(features), len(features[0])))
for i in range(len(features[0])):
	feat_rank = []
	for j in range(len(features)):
		feat_rank.append((feat_res[feat_ids[j][i]], j))
	feat_rank.sort()
	for j in range(len(features)):
		for k in range(len(features)):
			if feat_rank[k][0] == feat_res[feat_ids[j][i]]:
				ranking[j][i] = k + 1
				break

with np.printoptions(formatter={'float': '{:0.1f}'.format}):
	print(np.mean(ranking, 1))
