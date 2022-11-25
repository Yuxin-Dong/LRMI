import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# data = np.loadtxt('dataset/waveform/waveform-+noise.data', dtype=np.float64, delimiter=',')
# features = [
# 	[6, 10, 15, 9, 12, 11, 30, 37, 23, 29, ], # MRMI
# 	[6, 10, 15, 9, 5, 8, 4, 16, 12, 11, ], # LRMI
# 	[6, 10, 16, 33, 28, 39, 36, 21, 23, 31, ], # MIFS
# 	[6, 10, 16, 11, 9, 12, 23, 32, 30, 22, ], # FOU
# 	[6, 10, 15, 11, 16, 4, 9, 5, 8, 12, ], # MRMR
# 	[6, 10, 14, 15, 4, 9, 5, 16, 11, 7, ], # JMI
# 	[6, 10, 16, 14, 9, 11, 4, 8, 12, 15, ], # CMIM
# 	[6, 10, 14, 15, 4, 9, 5, 16, 11, 7, ], # DISR
# ]
# data = np.loadtxt('dataset/spambase/spambase.data', dtype=np.float64, delimiter=',')
# features = [
# 	[24, 25, 28, 26, 41, 40, 29, 43, 27, 45, ], # MRMI
# 	[20, 52, 6, 51, 4, 22, 15, 56, 23, 7, ], # LRMI
# 	[56, 26, 6, 3, 40, 46, 37, 41, 21, 43, ], # MIFS
# 	[56, 18, 54, 49, 24, 11, 48, 44, 36, 2, ], # FOU
# 	[56, 3, 6, 26, 51, 23, 52, 24, 15, 22, ], # MRMR
# 	[56, 18, 51, 52, 54, 6, 24, 55, 20, 15, ], # JMI
# 	[56, 18, 51, 52, 6, 20, 15, 55, 24, 26, ], # CMIM
# 	[6, 23, 26, 52, 15, 22, 24, 51, 25, 19, ], # DISR
# ]
# data = np.loadtxt('dataset/krvskp/krvskp.txt', dtype=np.float64)
# features = [
# 	[20, 9, 32, 31, 5, 34, 14, 0, 33, 6, ], # MRMI
# 	[20, 9, 32, 31, 5, 34, 14, 0, 33, 13, ], # LRMI
# 	[20, 9, 32, 31, 15, 8, 2, 27, 24, 11, ], # MIFS
# 	[20, 9, 32, 31, 14, 0, 33, 8, 1, 15, ], # FOU
# 	[20, 9, 32, 31, 14, 7, 15, 17, 5, 26, ], # MRMR
# 	[20, 9, 32, 31, 14, 7, 6, 15, 17, 5, ], # JMI
# 	[20, 9, 32, 31, 14, 7, 15, 5, 17, 21, ], # CMIM
# 	[20, 9, 32, 31, 28, 15, 13, 7, 26, 14, ], # DISR
# ]
data = np.loadtxt('dataset/breast/wdbc.txt', dtype=np.float64) # breast
features = [
	[27, 20, 21, 7, 10, 22, 29, 26, 1, 23, ], # MRMI
	[27, 20, 21, 7, 10, 22, 29, 26, 1, 23, ], # LRMI
	[22, 27, 1, 28, 19, 11, 18, 4, 14, 13, ], # MIFS
	[22, 24, 9, 14, 29, 3, 19, 18, 4, 11, ], # FOU
	[22, 24, 7, 1, 13, 27, 28, 23, 26, 10, ], # MRMR
	[22, 24, 23, 27, 7, 20, 13, 26, 3, 6, ], # JMI
	[22, 24, 27, 21, 7, 9, 13, 26, 1, 3, ], # CMIM
	[22, 20, 23, 3, 2, 0, 27, 7, 6, 26, ], # DISR
]

X = data[:, :-1]
y = data[:, -1]

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

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

classifier = SVC(class_weight='balanced')

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
