#coding: utf8 #Had to put that in here to be able to paste hebrew string literals (the functino validate_window_tav)

#Geniza data loading
import pandas as pd
import csv
import sys
sys.path.insert(0, '/Users/local_admin/Desktop/run_pg_tav/data')
import geniza

#Geniza data prep and stats
import math
import numpy as np
import matplotlib.pyplot as plt

#TAV model
from ja_transliteration_tool.run.e2e_pipe import Import, PipelineManager
import cProfile


#
################################
# Geniza data load, prep & stats
################################
#


def read_ja_articles():

	#pgpids (of interest)
	ja_articles_df = pd.read_csv("run_pg_tav/data/ja_articles_pgpids.csv")
	pgpid_df = ja_articles_df[["pgpid"]]

	#pgpids-contents (all)
	fnotes_df = pd.read_csv("run_pg_tav/data/pgp-metadata-main/data/footnotes.csv")
	ids_contents_df = fnotes_df[["document_id", "content"]]
	ids_contents_df.reset_index()

	return pgpid_df, ids_contents_df


#Method I (via looping) ===> 2.5 seconds (avg of 7 runs)
def process_ja_articles_looping(pgpid_df, ids_contents_df):

	ids_text = []
	unused_short = 0
	pgpids = set(pgpid_df["pgpid"].tolist())

	for idx, row in ids_contents_df.iterrows():

		content = str(row['content'])
		doc_id_str = row['document_id']
		if not math.isnan(doc_id_str) and int(doc_id_str) in pgpids:
			content = str(row['content'])
			if len(content) <= 5:
				unused_short = unused_short + 1
			else:
				ids_text.append([int(doc_id_str), content])
	
	return ids_text, unused_short


#Method II (via joining and filtering) ===> 2.1 seconds (avg of 7 runs)
def process_ja_articles_merging(pgpid_df, ids_contents_df):

	all_ids_text = pgpid_df.merge(ids_contents_df, left_on='pgpid', right_on='document_id')[['pgpid', 'content']]
	# ja_docs_stats(all_ids_text)
	# mask = (all_ids_text['content'].str.len() < 512) & (all_ids_text['content'].str.len() > 5)
	mask = (all_ids_text['content'].str.len() > 5)
	ids_text_df = all_ids_text.loc[mask]
	ids_text = ids_text_df.values.tolist()
	skipped = len(all_ids_text) - len(ids_text_df)
	return ids_text, skipped


def ja_docs_stats(ids_contents_df):

	articles_lens = ids_contents_df["content"].str.len()
	print(f"Total count is {len(articles_lens)}")
	print(f"Average is {np.nanmean(articles_lens)}")
	print(f"Variance is {np.var(articles_lens)}")
	binwidth = 500
	plt.hist(articles_lens, bins=range(0, 8000, binwidth))
	plt.xticks(range(0, 8000, binwidth))
	plt.xlabel("Documents length (# characters)")
	plt.ylabel("Documents count")
	plt.show()


def save_ja_articles(ids_text, skipped):

	# print(f"Running the model on {len(ids_text)} articles (skipping total of {skipped} articles)")
	with open('run_pg_tav/data/idd_ja_articles.csv', 'w') as f:
	    write = csv.writer(f)
	    write.writerow(['pgpid', 'content'])
	    write.writerows(ids_text)


def prepare_data(looping: False, save: True):

	pgpid_df, ids_contents_df = read_ja_articles()
	ids_text, skipped = process_ja_articles_looping(pgpid_df, ids_contents_df) if looping \
					else process_ja_articles_merging(pgpid_df, ids_contents_df)
	save_ja_articles(ids_text, skipped)
	return ids_text


def content_by_pgps(pgpids):

	ids_texts_df = pd.read_csv("run_pg_tav/data/idd_ja_articles.csv")
	return ids_texts_df[ids_texts_df['pgpid'].isin(pgpids)].values.tolist()


#
##########################################
# Sliding window (to handle long articles)
##########################################
#


class FORMATS:
	
	TEXT_GREEN = '\033[92m'
	BG_GREEN = '\033[42m'
	ENDF = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


def sliding_window(sequence, target_window = 300, ctxt_window = 100):

    for i in range(0, len(sequence), target_window):
    	
    	#chunk-based indexing (marking the index of the last character in the leading context and the target windows)
    	leading = ctxt_window if i > ctxt_window else 0
    	target = ctxt_window + target_window if i > ctxt_window else target_window
    	
    	yield [sequence[max(0, i - ctxt_window): i + target_window + ctxt_window], leading, target]


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


def run_sliding_window(pgpids, contents, target_window, ctxt_window, stich_back = True):

	chunked_articles = []
	for article_content, article_pgpid in zip(contents, pgpids):

		which_target_window = target_window if len(article_content) >= 512 else len(article_content)
		which_ctxt_window = ctxt_window if len(article_content) >= 512 else 0

		chunks_gen = sliding_window(article_content, target_window = which_target_window, ctxt_window = which_ctxt_window)
		chunks = list(chunks_gen)

		chunked_articles = chunked_articles + [geniza.GenizaArticle(original_text = chunk[0], \
									pgpid = article_pgpid, \
									ctxt_win_size = which_ctxt_window,
									target_win_size = which_target_window,
									original_leading_boarder = chunk[1], \
									original_target_boarder = chunk[2]) for chunk in chunks]

	run_tav_model_v2(chunked_articles, stich_back = stich_back)


#
###########
# TAV model
###########
#


def run_tav_model_v2(geniza_articles, stich_back = True):

	initial_input = Import()
	# initial_input.by_list_str(ids_text)
	# initial_input.by_idd_list_str(ids_text)
	initial_input.by_list_objects(geniza_articles)

	pm = PipelineManager(initial_input.output(), stich_back=stich_back)
	print(f"Your transliteration is ready! Please visit: {pm.output()}")


def run_tav_model(ids_text):

	initial_input = Import()
	# initial_input.by_list_str(ids_text)
	initial_input.by_idd_list_str(ids_text)

	pm = PipelineManager(initial_input.output())
	print(f"Your transliteration is ready! Please visit: {pm.output()}")


def validate_realtime_tav():

	one_letter_article = 'geniza.GenizaArticle(original_text = "א", pgpid = 0, ctxt_win_size = 0, target_win_size = 1, original_leading_boarder = -1, original_target_boarder = -1)'
	#4.85, 5.31, 4.92

	two_words_article = 'geniza.GenizaArticle(original_text = "אנה ולי דלך", pgpid = 0, ctxt_win_size = 0, target_win_size = 1, original_leading_boarder = -1, original_target_boarder = -1)'
	#5.24, 6.67, 6.62

	one_sentence_article = 'geniza.GenizaArticle(original_text = "אנה ולי דלך ואלקאדר עליה אן שא אללה אכי בל סיידי יום אלב לכד יומא כלון מן שהר חשון סנה תתסד ערפנא אללה ברכאת לנא ולכל ישראל כנת כרגת מן ענדכם ולם אלתקי בך וצעב עלי דלך כתיר ווצלת סאלם אלי אל", pgpid = 0, ctxt_win_size = 0, target_win_size = 1, original_leading_boarder = -1, original_target_boarder = -1)'
	#4.85, 5.69, 6.06

	cProfile.run(f"run_tav_model_v2([{one_sentence_article}], stich_back=False)")


def validate_window_tav():

	initial_input = Import()

	# text = [row[1] for row in ids_text]
	#print(text[:2])
	# initial_input.by_list_str(text[:2]) 

	# print(ids_text[:2])
	# initial_input.by_idd_list_str(ids_text[:10]) 

	s = "אנה ולי דלך ואלקאדר עליה אן שא אללה אכי בל סיידי יום אלב לכד יומא כלון מן שהר חשון סנה תתסד ערפנא אללה ברכאת לנא ולכל ישראל כנת כרגת מן ענדכם ולם אלתקי בך וצעב עלי דלך כתיר ווצלת סאלם אלי אל אסכנדרייה וכנת מעוול עלי רגועי אלי ענדכם תם טלעת אלי מצר וגדת מכלוף בן //משה יערף// עינין שרה אתחדתת מעה עלי אמר סכן בלאד אלהנד פערפני רסתאקהא וטלע פי באלי ללוקת אן אטלע אלי אלהנד ולם נקרא משלי ועיני כסיל בקצה ארץ אלי אנה כרגת מן מצר אכר איאר סנה תתסג ובלגת אלי קוץ בעהד שהרין אלא וטלעת פי אלצחרא אקמנא פיהא איאם תם אנה וצלת אלי עידאב בלחקיק אנהא בלד אלעדאב תם אנה כרגנא מקלעין פי מרכב מא פיהא מסמאר חדיד אלא מרבוטא באלחבאל ואללה יסתר בסתרה וצלנא אלי בלד יקאל להא סואכן הי באל חקיק אכץ מואכן תם אנה וצלנא אלי בלד יקאל להא באצע כי כשמה כן היא הי אמר אכץ ואשר אלמואצע תם אנה וצלנא אלי בלד יקאל להא דהלך יקאל עליהא אלמתל ואת עלית על כלנה הי בלד מהלך ועלי כל חאל מא יצייענא אלרב עז וגל ברחמתה ובדי לי כמא אלתקית מע אברהם בן עמי עלוש מא כאן נאכד מנה אלקליל מרגאן אלדי כאן לך מעה כאן יכון תדכאר ועלי כל חאל בחק הדה אלאחרף עליך ובחק אלאהבה ואלכבז ואלכאס אלדי שרבנאה ואכלנאה אן לא תכליני מן אלדעא לא אנת ולא ואלדתך ולא כאלך יעקב ואנא אעלם אנך דאעי לנאס אגמעין אחרי ואוכד חביבך ואללה תע׳ ילטף ברחמתה ואנא יאסיידי עלי ראס תעדיה אלבחר אלכביר ליס הו בחר אטראבלס ומא נעלם אן כאן נלתקי אם לא נחב מן אללה ומנך אן תדעו לי ותחאללני אנת וכאלך יעקב אן חסאב ואכד //אלי// כתב אלקי לי באטראבלס סוי אנת והו סוד אמונה אומן לא יקראה אחד ולא יערף איצא בה לא הו ולא גירה ואללה יסאמחך ויחאללך במא גרי ביננא אנת וכאלך יעקב דניא ואכרא ותגעל ואלדתך אן תבלג עני אכואתי אלסלאם ותגעלהם ידעו לי ולו קדרת יאסיידי אן אדפע קק דינאר ויקים אלמרכב אלי אן יתוטא אלבחר לפעלת וחסבי ותקתי באללה וחדה ותבלג עני לאך אלשקיק אבו אסמאעיל ואכוך אלסלאם ותם תחלפה אן לא יכליני מן אלדעא ותבלג עני מר ור זכריה אלסלאם ואולאדה אחיאהם אללה אלסלם ומשה בן עמי ואכותה אלסלאם וכאלי יחיי ועלוש בן אברהם תבלגהם סלאם וסאעה אן יקראו? כתאבי הדא תכרקה אללה אללה ואלעבד יקבל יד מולאי וכנת נסית אן אכד סדר מועד מן אלכניסיה אן כנת מחתאג אליה תתכלף בה ואן כאן ראית אן יצלח יסכן פי אלדאר ויצלח . . . אן יכון פיהא פעלת דלך אללה  יכאפיך ותשארך ראייך פי דלך מע אכתי ואכי וכאלך יעקב ותבלג כלפה בן כתני ובקא אלסלם וכדלך אולאד אכתי אלסלם וחסבי ותקתי אללה"
	s2 = "תקדמת כתבי לחצרה מולאי אלשיך אלאגל אטאל אללה בקאה ואדאם תאיידה ועלאה ותמכינה וכבת אעדאה עדה ואכרהא שדה פי כרקה צחבה אלשיך אבו אלפצל בן מעלא וכתאב כתאבה מע אלפיג מן בן צגיר ואכרהא מע אבו אלימן בן צגיר ומעה ה̇ דנא' עדד וזנהא ד̇ ונצף ושרחת לה חאלהא וסבב הדה אלאחרוף אן אגתמעו מעאמלי ואתפקו אן מאלי שלל ירא יי עליהם וישפוט לו ⟦שר⟧ שרחת לה מא גרי לי מע אלשיך הבה אלחמוי לטאל שרחה וה[ק]בה לא יבקי לי חק לא ענדה ולא ענד גירה סִפרתה ש̇פ̇ה̇ וסאפר לרוח ק̇פ̇ ואכדת לה מן יוסף בן עטא ק̇ דינ' ומן עלוש ק̇ דינ' ומן בן חארת ואבו אלפצל בן צגיר ק̇ק̇ באע אללאך ל̇ דינ' דפע סער כ̇ה̇ וגא כל בהאר ש̇ע̇ה̇ מד[...] ולם ידפע לאחד שי אלא בעץ̇ ראס מאלהם ובעץ̇ דון וגאב לי אנא ש̇כ̇ וגרי לי מעה גנאיז קדרת(?) אלנעוש ואלאזקאק ואלסואד עליהאוס[ל]ב אלבצר ומא כלית לא צביאן ודפעת לדיין דינר ללעניים חתי יסתחלפו לי יום יז בתמוז פי אלגמע וכאן גמעאן עטים ומא בקי אלא ימיני וחלף אלף ימין לא צלחני ואנדר ללעניים דינר אן תמנע מן אלימין וקרא עליה אלדיין גלול מא א[ו]גב מתלה גירי קט וקרא עליה אלדיין סורו נא ואכדת ספר תורה מן יד אל[דיין] וגית אליה נסלמה לה רגעת אלגמאעה אליה אכדוה קאלו לה הדא גילגול מא סמע קט מתלה וחאלוני עלי כ̇ה̇ דינר פתסהלת פי [א]כדהא לסואלהם חשמה מנהם ואלפי אלנאס אנה מא יעתקד ימין לאן ח]לף לא צלחני וקרי עליה אלגלול וסורו נא כליתה מע אלל[...]"
	s3 = "[...]ם אכוךּ [...]ב אמארה̈ כל [...]כה וגרם אלף וכ̇מס מאיה̈ דרהם ומן גהה̈ אלעבד מבארך פשעיה פי אסכנדריא ואכ̇דה אלמחתסב בלאש מא סלם פיה שי וג̇רם אכ̇וך כמס מאיה לכון אנאס קאלו לה הדא גוי פמא קעד אכ̇וך פי מצר בעד מא גא מן אסכנדריא אלא נצף שהר תופא סלימאן ר̇י̇ת̇ פחזנא עליה חזן עט̇ים מנהא וסאפרנאלטור ומא עמרה אשתרי חואיג וקמאש ועקאראת [ו]שראבאת ונחאס מסתעמל וצנאדיק ובצת ולא בקא [ש]י חתי אשתרי מנה וד̇הב כת̇יר כאן מעה שי לה ושי ודאעה [אל]די חמﹼל מן מצר כ̇מסה ועשרין גמל פלמא וצלנא [א]לטור וצלת ורקה מן ענד אלחג רשיד אל[...]י באן [א]כוך מא יסאפר אלא פי מרכבה פאספר[נא...פת]גאוזנא אלפוי וטיר אלפלאח פסאפרנא בעדהם בעשרין יוםמשינא תלאתה איאם יום אלראבע חצל עלינא רוח עאסף רדינא אלי מוצ̇ע מא כנﹼא פקעדנא ארבעה איאם נרוח ונרג̇ע פאדא יום אלר[אב]ע ונחנא ראיחין נג̇רק פאדא בד̇כ̇אן טלע מן אלמרכב פי סאעה טלעת אלנאר אחתרקת אלקלוע וכל מא פי אלמרכב רגאל אלמרכב יערפו יעומו רמו נפוסהם אלי אלבחר מא בקא אלא מן לא יערף יעום ומן גהת אכ̇וך באנה נזל אל[י] אלבליג ומא אעלם מא אכ̇ר פי חזתה ותחזם בתובין תלאת̇ה פלמן קרבת אלינא אלנאר רמית רוחי אלבחר קאלת בדאל מא ארוח חריק ארוח גריק רמית רוחי אלבחר ולם אערף מא גרי לאכ̇וך לקית חבל פי אלמרכב תעלקת בה קעת מעלק אלי אן אכ̇תלעו אכתאפי וכאנת אלנאר תקע פוקי מן פוק אלמרכב ואנא גאטס פי אלבחר וכל מוגי כמתל אלגבל פוקי וד̇אך אלמוצ̇ע אלדי וקענא פיה מא לה קראר קאלו אלרבאבין אכת̇ר מן אלפין קאמה פי מוצ̇ע יקאל לה תאראן פלמן חסית אכת̇אפי אנכלעו || מני פלת אלחבל וגטסת יגי קאמתין וטלעת וגטסת תאניה וטלעת ולם יכון ענדי שי אתעלק בה ולא אמסכה פי טלועי צכ̇ר אללה לי כשבה ולם יכון ענדי שי אלא מא אערף מניין גתני בעת לי איאהא הק̇ב̇ה̇ פי סאעה גרק אלמרכב לם אדרי אלא בגמאעה נסא ורגאל קדר ארבעין רוח והם עלי אכשאב ורבטו אלכ̇שב כלהם בעצ̇הם בעץ̇ ובקינא כלנא פי מוצ̇ע ואחד קעדנא כדא עלי אלכשב יומין ולילא לם אדרי יום אלתאני אלא ואכ̇וך מענא עלי אלכשב והו עריאן כאדם והו ביטאלע פי אלרוח ואנא אלאכ̇ר  כנת עריאן מתלה מא נדרי יום אלתאלת אלא וגאנא צנבוק אלמרכב אכ̇דנא וטלע בנא אלי אלבר וקד נחנא מותא מן אלגוע ואלעטש ואלברד ומן גהה̈ אלגוים אלדי טלעו מענא אלי אלבר מא אחד וקף אלא ראח יפתש עלי אלמא לם בקי פי אלבר אלא אנא ואכ̇וך וואחד גוי פשפקו אהל אלצנבוק עלי אכ̇וך וכאן פי מ/ו/צ̇ע אכ̇ר מרכב ראיח אלי אלטו{ר} וקאלו נטלע במוסי אלי אלמרכב ונסקיה שרבה̈ מא ונגטיה בכ̇רקה לאלא ימות ולא ילקא מן ידפנה ואמא אנא מא כנת אערף אלליל מן אלנהאר מן אלוגע ואלפגעה פרקת דיך אללילא אנא ואלגוי פי אלבר ולם מענא שי לא נאכל ולא נשרב ואנא עריאן כאדם אללה עליא שאהד מא וצלת ללבר ולא עלי גסמי כ̇רקה ולו אנה בפלס אלא עריאן כמתל אלצגיר ומא זאד קתלני אלא אלברד פלמא אצבח אלצבח לקית אלגוי מיית פבקית וחדי לא אין אערף לא ארוח ולא אגי פבקית אמשי סאעה ואקעד סאעה מן אלתעב משיא אלי נצף אלנהאר עלי את̇ר אלנאס חתי אללה אוצלני אלי אלגמאעה פטלבת מנהם שרבת אלמא כאן אלחדית יום אלראבע צאבר מן אלמא ואלטעאם חתי לקית ענדהם קליל מא מאלח יקטע אלחלק מא אדרי אלא וגונאס מן אלמרכב אלדי טלעו בה אכ̇וך ודכרו באן סאעה̈ טלועה אלמרכב תופא פאגו נאס אכ̇רין ואכ̇רין והם יקולועלי הדה אלכלאם פמשית מע אלגמאעה תמאן איאם חתי וצלנא ללטור פרחמני אללה ברבאן קאת לי ומקדם אלבלד וכאן מעהם גמל פעקבוני סאעה וסאעה ואכתלם גסמי כֻלה מן אלשמס וכאן כֻלה לחמא ואחדא פלמא וצלת אלי אלטור ואנא עריאן פאעטאני ואחד נצראני כלק גבא פקעת ענדה תלאתה איאם ראש השנה אלסבת ואלאחד ויום אלאתנין באלראס פאכתרית מן ואחד ערבי בתלתין דרהם אלי מצר פיקעד בי יומין אלסבת ואלכפור אלי אן וצל בי אלקאהרה וקד אנא מיית מן אלתעב וגסמי מכלץ כולה מן אלשמס וצלת אלי אלכניס קאמו אליהוד גבו ללערבי תלאתין דרהם כראה ורמו עליא עמאמה וקמיץ לאני מא וצלת להם אלא עריאן מא עליא אלא כלק גבת אלנצרה ואנא אקרע חאפי פקעת פי אלקאהרה עשרין יום פאתנין תלאתה אחסנו אליא ורח{ת} אלי מצר ודא אלוקת אנא קאעד פי מצר לא אערף אין ארוח ולא אין אגי ולא מעי שי כנת אתעלק מע אחד מן אצחאב אלמראכב ואללה עליא שאהד לם טלעת מן אלבחר ומעי לא מא אקל ולא מא אכתר ול[[א]] אנה אנה כרקא תסוא פלס ולא אערף מא יכון אמרי וכיף יכון כלאצי אריד מן אללה תע ומן הקבה אלדי אנשלני מן אלבחאר ואלנאר קאדר אן יקאבל מא ביני ובינכם פי כיר ועאפיה ואללה קאדר עלי כל שי ואלשכץ אלדי כנת מתכי עליה ראח אלי רחמת אללה תע מא בקי בלחין אלא הקבה ולא תסאל מא חזנת עליה ועלי סלימאן וחק דיני לו אחכי לך בכל מא גרי לנא מן יום מא סאפרנא אלי אן גרי עלינא הדה אלשדה מא כאן יכפאני בקדר הדה אלכתאב עשר מראר אלי כם הדא אכתצאר ולא פי מצר אחד אלא והו חזין עליה יהוד ומסלמין וגרקו גוים כתיר וחריק כתיר מא אחתרק ועליהם מתלמא(?) אחתרקו על אכוך ולא טלע מרכב תגאר ושחנא(?) מתל הדה אלמרכב לקל אלקסרייה(?) ואלעוון ואלדי כאן תחת א . . . שי יסוי אלפין וכמס מאיה דרהם לי ולגירי ודאיע אלא אלאמר ללא תע פלא אחד בידה חילה אללה עטא ואללה אכד ומא כאן אמתנאעי באני אתעלק מע אחד ואנא עריאן ואגי אליכם אלא ינכסר קלבכם כיף אני גית אנא ואכוך ולא אלא ינסא(?) יהון אלצ . . . . . . . . . מלך מלכי המלכים הקבה בלוצול אליכם ואלסלם ואלעאפיא ובאחסאנך סלם לי עלי ואלדך ועלי אהל אלבית כלהם צגיר וכביר בעד אן תכץ"

	# s = s2
	s = s3

	#A. 1 512
	# initial_input.by_idd_list_str([[0, s[:512]]])

	#B. 2 256's
	# initial_input.by_idd_list_str([[0, s[:256]], [1, s[256: 512]]])

	#C. 4 128's
	initial_input.by_idd_list_str([[0, s[:128]], [1, s[128: 256]], [2, s[256: 384]], [3, s[384: 512]]])

	pm = PipelineManager(initial_input.output())
	print(f"Your transliteration is ready! Please visit: {pm.output()}")


def transliterate_geniza():

	#
	#Test sliding window
	#
	# test_sliding_window(article_content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", \
	# 					target_window = 3, ctxt_window = 1)
	# test_sliding_window(article_content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", target_window = 5, ctxt_window = 0)

	#
	#Run sliding window (two articles)
	#
	# ids_texts = prepare_data(looping = False, save = True)
	# run_sliding_window(contents=[ids_texts[7][1], ids_texts[4][1]], \
	# 				   pgpids = [ids_texts[7][0], ids_texts[4][0]], \
	# 				   target_window = 300, \
	# 				   ctxt_window = 100, \
	# 				   stich_back = False)

	#
	#Run sliding window (two articles)
	#
	# ids_texts = prepare_data(looping = False, save = True)
	# run_sliding_window(contents=[ids_texts[9][1], ids_texts[6][1]], \
	# 				   pgpids = [ids_texts[9][0], ids_texts[6][0]], \
	# 				   target_window = 300, \
	# 				   ctxt_window = 100, \
	# 				   stich_back = False)


	ids_texts = content_by_pgps([444, 4268])
	run_sliding_window(contents=[ids_texts[0][1], ids_texts[1][1]], \
					   pgpids = [ids_texts[0][0], ids_texts[1][0]], \
					   target_window = 300, \
					   ctxt_window = 100, \
					   stich_back = True)

	#
	#Run sliding window (all articles)
	#
	# ids_texts = prepare_data(looping = False, save = True)
	# ids, contents = zip(*ids_texts)
	# run_sliding_window(contents = contents, \
	# 					pgpids = ids, \
	# 					target_window = 300, \
	# 					ctxt_window = 100, \
	# 					stich_back = True)


def main(): 

	transliterate_geniza()
 

if __name__=="__main__": 

	main()












