#coding: utf8


import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import statistics as st


DATA_DIR = "../resources/pgp_data"


def read_ja_articles():

	#pgpids (of interest)
	ja_articles_df = pd.read_csv(f"{DATA_DIR}/ja_articles_pgpids.csv")
	pgpid_df = ja_articles_df[["pgpid"]]

	#pgpids-contents (all)
	fnotes_df = pd.read_csv(f"{DATA_DIR}/footnotes.csv")
	ids_contents_df = fnotes_df[["document_id", "content"]]
	ids_contents_df.reset_index()

	return pgpid_df, ids_contents_df


def process_ja_articles_merging(pgpid_df, ids_contents_df):

	all_ids_text = pgpid_df.merge(ids_contents_df, left_on='pgpid', right_on='document_id')[['pgpid', 'content']]
	mask = (all_ids_text['content'].str.len() > 5)
	ids_text_df = all_ids_text.loc[mask]
	ids_text = ids_text_df.values.tolist()
	skipped = len(all_ids_text) - len(ids_text_df)
	return ids_text, skipped


def ja_docs_stats(ids_contents):

	articles_lens = [len(doc[1]) for doc in ids_contents]
	print(f"Total count is {len(articles_lens)}")
	print(f"Average document length is {round(np.nanmean(articles_lens), 2)}")
	print(f"Variance of document length is {round(np.var(articles_lens), 2)}")
	binwidth = 500
	plt.hist(articles_lens, bins=range(0, 8000, binwidth))
	plt.xticks(range(0, 8000, binwidth))
	plt.axvline(st.mean(articles_lens), color='k', linestyle='dashed', linewidth=1)
	plt.xlabel("Documents length (# characters)")
	plt.ylabel("Documents count")
	plt.xticks(rotation=30)
	plt.savefig(f'{DATA_DIR}/pgp_ja_data_stats.png', bbox_inches="tight")


def save_ja_articles(ids_text, skipped):

	with open(f'{DATA_DIR}/idd_ja_articles.csv', 'w') as f:
		write = csv.writer(f)
		write.writerow(['pgpid', 'content'])
		write.writerows(ids_text)


def prepare_data(save: True):

	pgpid_df, ids_contents_df = read_ja_articles()
	ids_text, skipped = process_ja_articles_merging(pgpid_df, ids_contents_df)
	if save:
		save_ja_articles(ids_text, skipped)
	return ids_text


def content_by_pgps(pgpids):

	ids_texts_df = pd.read_csv(f"{DATA_DIR}/idd_ja_articles.csv")
	return ids_texts_df[ids_texts_df['pgpid'].isin(pgpids)].values.tolist()

def prep_and_stats():
	id_texts = prepare_data(save = True)
	ja_docs_stats(id_texts)


# def main():
# 	prep_and_stats()
#
#
# if __name__=="__main__":
# 	main()
