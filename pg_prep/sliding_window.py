#coding: utf8

from pg_prep.pgp_record import GenizaArticle
import cProfile

class FORMATS:
	
	TEXT_GREEN = '\033[92m'
	BG_GREEN = '\033[42m'
	ENDF = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


def sliding_window(sequence, target_window=300, ctxt_window=100):
	for i in range(0, len(sequence), target_window):
		# chunk-based indexing (marking the index of the last character in the leading context and the target windows)
		leading_barrier = ctxt_window if i > ctxt_window else 0
		target_barrier = ctxt_window + target_window if i > ctxt_window else target_window
		yield [sequence[max(0, i - ctxt_window): i + target_window + ctxt_window], leading_barrier, target_barrier]


def show_chunks(article, chunks):

	print(f"Whole sequence: |{article}| = {len(article)}")
	i=1
	for chunk in chunks:
		print(f"Chunk {i}: |{chunk[0][: chunk[1]]}{FORMATS.BOLD}{FORMATS.BG_GREEN}{chunk[0][chunk[1]: chunk[2]]}{FORMATS.ENDF}{chunk[0][chunk[2]:]}| = {len(chunk[0])}")
		i = i + 1


def test_sliding_window(article_content, target_window, ctxt_window):

	print(f"Target window: {target_window} while context window: {ctxt_window}")

	chunks = sliding_window(article_content, target_window = target_window, ctxt_window = ctxt_window)
	show_chunks(article_content, chunks)


def slice(pgpids, contents, target_window, ctxt_window):

	chunked_articles = []
	for article_content, article_pgpid in zip(contents, pgpids):

		which_target_window = target_window if len(article_content) >= 512 else len(article_content)
		which_ctxt_window = ctxt_window if len(article_content) >= 512 else 0

		chunks_gen = sliding_window(article_content, target_window = which_target_window, ctxt_window = which_ctxt_window)
		chunks = list(chunks_gen)

		chunked_articles = chunked_articles + [GenizaArticle(original_text = chunk[0],
									pgpid = article_pgpid,
									ctxt_win_size = which_ctxt_window,
									target_win_size = which_target_window,
									original_leading_boarder = chunk[1],
									original_target_boarder = chunk[2]) for chunk in chunks]
	return chunked_articles

