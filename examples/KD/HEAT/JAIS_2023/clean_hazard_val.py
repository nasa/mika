import pandas as pd


full_val = pd.read_csv(r"C:\Users\srandrad\smart_nlp\examples\KD\HEAT\JAIS_2023\hazard_validation.csv", index_col=0)
test = pd.read_csv(r"C:\Users\srandrad\smart_nlp\examples\KD\HEAT\JAIS_2023\labeled_hazards_test.csv", index_col=0)
val = pd.read_csv(r"C:\Users\srandrad\smart_nlp\examples\KD\HEAT\JAIS_2023\labeled_hazards_val.csv", index_col=0)

#cleans the full validation set to make it consistent with the val and test set
full_unordered = pd.concat([val,test])
sorter = full_val['Tracking #'].tolist()
sorter_index = dict(zip(sorter, range(len(sorter))))
full_unordered['id_ind'] = full_unordered['Tracking #'].map(sorter_index)
ordered = full_unordered.sort_values(['id_ind'], ascending=True)
ordered = ordered.drop(["id_ind"], axis=1).reset_index(drop=True)
ordered.to_csv(r"C:\Users\srandrad\smart_nlp\examples\KD\HEAT\JAIS_2023\hazard_validation.csv")
print(ordered)
print(full_val)

#cleans the val and test set to make it consistent with the full validation set
# for i in range(len(full_val)):
#     id = full_val.at[i,"Tracking #"]
#     correct_slice = full_val.iloc[i][:]
#     if id in val['Tracking #'].tolist():
#         ind = val.loc[val['Tracking #']==id].index[0]
#         for col in full_val.columns:
#             val.at[ind, col] = full_val.at[i, col]
#     elif id in test['Tracking #'].tolist():
#         ind = test.loc[test['Tracking #']==id].index[0]
#         for col in full_val.columns:
#             test.at[ind, col] = full_val.at[i, col]

# test.to_csv(r"C:\Users\srandrad\smart_nlp\examples\KD\HEAT\JAIS_2023\labeled_hazards_test.csv")
# val.to_csv(r"C:\Users\srandrad\smart_nlp\examples\KD\HEAT\JAIS_2023\labeled_hazards_val.csv")
