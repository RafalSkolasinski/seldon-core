#!/bin/bash

docker tag clean_text_transformer:0.1 rafalskolasinski/seldon-experiments:clean_text_transformer-0.1
docker tag data_downloader:0.1 rafalskolasinski/seldon-experiments:data_downloader-0.1
docker tag lr_text_classifier:0.1 rafalskolasinski/seldon-experiments:lr_text_classifier-0.1
docker tag spacy_tokenizer:0.1 rafalskolasinski/seldon-experiments:spacy_tokenizer-0.1
docker tag tfidf_vectorizer:0.1 rafalskolasinski/seldon-experiments:tfidf_vectorizer-0.1


docker push rafalskolasinski/seldon-experiments:clean_text_transformer-0.1
docker push rafalskolasinski/seldon-experiments:data_downloader-0.1
docker push rafalskolasinski/seldon-experiments:lr_text_classifier-0.1
docker push rafalskolasinski/seldon-experiments:spacy_tokenizer-0.1
docker push rafalskolasinski/seldon-experiments:tfidf_vectorizer-0.1

