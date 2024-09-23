# BERT based machine transliteration (Judaeo-Arabic to Arabic)

### Environment setup
```
%%setup
#@title Environment setup
pyenv virtualenv 3.8 pgp_transliteration
pyenv activate pgp_transliteration
pip install -r global_def/requirements.txt
huggingface-cli login --token <[hugging_face_token](https://huggingface.co/docs/hub/en/security-tokens)>
PYTHONPATH="<local_path_to_cloned_repo>/pgp_transliteration:$PYTHONPATH"
export PYTHONPATH
```