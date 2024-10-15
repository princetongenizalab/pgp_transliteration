
class GenizaArticle(object):

	_pgpid = -1

	_ctxt_win_size, _target_win_size = -1, -1

	_original_leading_boarder, _original_target_boarder = -1, -1
	_original_text, _original_leading_ctxt, _original_target, _original_trailing_ctxt = "", "", "", ""

	_processed_leading_boarder, _processed_target_boarder = -1, -1
	_processed_text, _processed_leading_ctxt, _processed_target, _processed_trailing_ctxt = "", "", "", ""

	_processed_words = []

	_leading_end_word, _target_end_word = -1, -1


	def __init__(self, pgpid, ctxt_win_size, target_win_size, original_text, original_leading_boarder = -1, original_target_boarder = -1):

		self._pgpid = pgpid
		self._original_text = original_text

		self._ctxt_win_size = ctxt_win_size
		self._target_win_size = target_win_size

		if original_leading_boarder > -1 and original_target_boarder > -1:
			self._original_leading_boarder = original_leading_boarder
			self._original_target_boarder = original_target_boarder

	
	def find_word_indices(self, which_boarders = 0):

		leading_boarder = self._original_leading_boarder if which_boarders == 0 else self._processed_leading_boarder
		target_boarder = self._original_target_boarder if which_boarders == 0 else self._processed_target_boarder

		word_idx, char_idx = 0, 0
		for word in self._processed_words:
			crt_word = word.original_word if which_boarders == 0 else word.processed_word
			char_idx = char_idx + len(crt_word) + 1 #for the space
			if self._leading_end_word > -1 and char_idx >= target_boarder:
				self._target_end_word = word_idx
				break
			elif self._leading_end_word == -1 and char_idx >= leading_boarder:
				self._leading_end_word = word_idx
			word_idx = word_idx + 1
		self._target_end_word = self._target_end_word if self._target_end_word > 0 else len(self._processed_words)
		self._target_end_word = self._target_end_word - 2 if self._ctxt_win_size > 0 else self._target_end_word - 1


	def find_text_indices(self, which_text, leading_end_word, target_end_word):

		leading_end_text, target_end_text = -1, -1
		char_idx, word_idx = 0, 0

		for word in self._processed_words:

			word_len = len(word.processed_word) if which_text else len(word.original_word)
			char_idx = char_idx + word_len + 1 #for the space

			if leading_end_word > -1 and word_idx == target_end_word:
				target_end_text = char_idx
				break
			elif word_idx == leading_end_word:
				leading_end_text = char_idx - word_len - 1

			word_idx = word_idx + 1

		if which_text:
			self._processed_leading_boarder = leading_end_text
			self._processed_target_boarder = target_end_text
		else:
			self._original_leading_boarder = leading_end_text
			self._original_target_boarder = target_end_text


	def mark_original_text(self):

		self._original_leading_ctxt = self._original_text[: self._original_leading_boarder]
		self._original_target = self._original_text[self._original_leading_boarder: self._original_target_boarder]
		self._original_trailing_ctxt = self._original_text[self._original_target_boarder: ]


	def adjust_original_text(self):

		self._original_text = ' '.join(word.original_word for word in self._processed_words)
		self.find_text_indices(False, self._leading_end_word, self._target_end_word)

		self.mark_original_text()


	def mark_processed_text(self):

		self._processed_leading_ctxt = self._processed_text[: self._processed_leading_boarder]
		self._processed_target = self._processed_text[self._processed_leading_boarder: self._processed_target_boarder]
		self._processed_trailing_ctxt = self._processed_text[self._processed_target_boarder: ]


	def adjust_processed_text(self):

		self._processed_text = ' '.join(word.processed_word for word in self._processed_words)
		self.find_text_indices(True, self._leading_end_word, self._target_end_word)

		self.mark_processed_text()


	def align_boarders(self, which_boarders = 0):

		self.find_word_indices(which_boarders = which_boarders)

		self.adjust_processed_text()
		self.adjust_original_text()


	def assign_processed(self, processed_words):
		
		self._processed_words = processed_words
		self.align_boarders(which_boarders = 0)


	def __repr__(self):
		return f"{self._processed_leading_ctxt}\n{self._processed_target}\n{self._processed_trailing_ctxt}"


	def substrings(self, s):
		for j in range(0, len(s)):
			yield s[0:j+1]


	def intersect(self, s1, s2, which_to_reverse):
		set1 = set([s[::-1] for s in list(self.substrings(s1))]) if which_to_reverse == 0 else set(self.substrings(s1))
		set2 = set([s[::-1] for s in list(self.substrings(s2))]) if which_to_reverse == 1 else set(self.substrings(s2))
		return set1 & set2


	def detect_and_fix_errors(self, prev_article):

		case_b_err = max(self.intersect(prev_article._processed_target[::-1], self._processed_target, 0), key=len, default="")
		if len(case_b_err) > 2:
			print(f"Case B error detected => {case_b_err}")
			#fixing the processed/output/Arabic
			self._processed_leading_boarder = self._processed_leading_boarder + len(case_b_err)
			self.mark_processed_text()
			#fixing the original/input/Judaeo-Arabic
			self._leading_end_word = self._leading_end_word + len(case_b_err.split())
			self.adjust_original_text()

		case_c_err = max(self.intersect(prev_article._processed_trailing_ctxt, self._processed_leading_ctxt[::-1], 1), key=len, default="")
		if len(case_c_err) > 2:
			print(f"Case C error detected => {case_c_err}")
			#fixing the processed/output/Arabic
			self._processed_leading_boarder = self._processed_leading_boarder - len(case_c_err)
			self.mark_processed_text()
			#fixing the original/input/Judaeo-Arabic
			self._leading_end_word = self._leading_end_word - len(case_c_err.split())
			self.adjust_original_text()


	def merge(self, another_article):

		if self._pgpid != another_article._pgpid:
			print(f"Can't merge two different articles!")
			return

		self._original_leading_ctxt, self._processed_leading_ctxt = "", ""
		self._original_target = self._original_target + another_article._original_target
		self._processed_target = self._processed_target + another_article._processed_target
		self._original_trailing_ctxt, self._processed_trailing_ctxt = "", ""
