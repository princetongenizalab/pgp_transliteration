
def present_output(output_format, tman):

    if output_format == "by_list_str":
        print("Your transliteration is ready! Here are the results:")
        for sentence in tman.output():
            print("JA input: ")
            print(sentence[0])
            print("Transliterated output: ")
            print(sentence[1])
            print()

    elif output_format == "by_docx_path":
        print(f"Your transliteration is ready! Please visit: {tman.output()}")


def transliterate_ja():

    output_format = "by_list_str"
    # output_format = "by_docx_path"
    ja_text = "חצרנא נחן אלשהוד"

    from pg_prep.pgp_record import GenizaArticle
    test_ja = [GenizaArticle(original_text=ja_text, pgpid=-1, ctxt_win_size=100, target_win_size=300)]

    from run.e2e_pipe import TransliterationMan
    tm = TransliterationMan(test_ja, output_format=output_format)
    present_output(output_format, tm)


def transliterate_pgp_ja():

    from pg_prep.sliding_window import test_sliding_window
    test_sliding_window(article_content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", target_window = 3, ctxt_window = 1)
    test_sliding_window(article_content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", target_window = 5, ctxt_window = 0)

    from pg_prep.prep_pg_data import content_by_pgps
    ids_texts = content_by_pgps([4268, 444])

    from pg_prep.sliding_window import slice
    sliced = slice(contents=[ids_texts[0][1], ids_texts[1][1]],
				   pgpids = [ids_texts[0][0], ids_texts[1][0]],
				   target_window = 300,
				   ctxt_window = 100)

    from run.e2e_pipe import TransliterationMan
    output_format = "by_docx_path"
    tm = TransliterationMan(sliced, output_format=output_format, stich_back=True)
    present_output(output_format, tm)


def main():

    # Sentence-level transliteration
    # transliterate_ja()

    # PG data pre-processing
    # from pg_prep.prep_pg_data import prep_and_stats
    # prep_and_stats()

    # Document-level transliteration
    transliterate_pgp_ja()

if __name__=="__main__":
    main()
