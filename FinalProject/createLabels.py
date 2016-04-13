import pandas as pd
bids=pd.read_csv("bids.csv")
train=pd.read_csv("train1.csv")

bids_ids=list( bids['bidder_id'])
train_outcomes=list(train['outcome'])
train_ids=list( train['bidder_id'])
out_list=[]
for id in bids_ids:
	j=0
	if id in train_ids:
		i=train_ids.index(id)
		out_list.append(train_outcomes[i])
		# bids['outcome'][j]=train_outcomes[i]
	else:
		out_list.append(2)
		# bids['outcome'][j]=2
	j+=1

new=pd.DataFrame({'outcome':out_list})
final=bids.join(new)

# print final
final.to_csv('bids_labelled.csv')


