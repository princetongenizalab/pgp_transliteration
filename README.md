# Document-level Machine transliteration

Running the code from a Bash command line console

### Setup the environment
```
git clone https://github.com/princetongenclearizalab/pgp_transliteration.git
cd pgp_transliteration
pyenv virtualenv 3.8 pgp_transliteration
pyenv activate pgp_transliteration
pip install -r global_def/requirements.txt
huggingface-cli login --token <[hugging_face_token](https://huggingface.co/docs/hub/en/security-tokens)>
PYTHONPATH="<local_path_to_cloned_repo>/pgp_transliteration:$PYTHONPATH"
export PYTHONPATH
```

### Prepare the input

Prepare a list of Judaeo-Arabic strings associated with IDs
```
    from pg_prep.prep_pg_data import content_by_pgps
    ids_texts = content_by_pgps([4268, 444])
```

Break-down long documents into smaller groups of interleaving text sequences. 

```
from pg_prep.sliding_window import slice
sliced = slice(contents=[ids_texts[0][1], ids_texts[1][1]],
                pgpids = [ids_texts[0][0], ids_texts[1][0]],
                target_window = 300,
                ctxt_window = 100)
```

### Invoke the Bert-based model
```
from run.e2e_pipe import TransliterationMan
output_format = "by_docx_path"
tm = TransliterationMan(sliced, output_format=output_format, stich_back=True)```

```
### And present the result

```
present_output(output_format, tm)
```