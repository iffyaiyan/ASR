import torch
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import numpy as np
import soundfile as sf


def load_custom_dataset(data_dir):


    data_files = {
    "train": f"{data_dir}/train.tsv",
    "test": f"{data_dir}/test.tsv",
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
     return dataset


def prepare_dataset(batch):


    audio_array, sampling_rate = sf.read(batch["audio_path"])
    batch["input_values"] = processor(audio_array, sampling_rate=sampling_rate).input_values[0]
    batch["input_length"] = len(batch["input_values"])

     with processor.as_target_processor():
        batch["labels"] = processor(batch["transcript"]).input_ids
     return batch


def compute_metrics(pred):


    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
     return {"wer": wer}


def main():

    # Load dataset
    dataset = load_custom_dataset("common_voice_dataset")

    # Initialize tokenizer and feature extractor
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

    # Create processor (combines tokenizer and feature extractor)
     global processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Preprocess dataset
    dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=4)

    # Load metric
     global wer_metric
    wer_metric = load_metric("wer")

    # Load pre-trained model
    model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    )

    # Define training arguments
    training_args = TrainingArguments(
    output_dir="./wav2vec2-court-transcription",
    group_by_length=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    )

    # Create Trainer instance
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics
   )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("./wav2vec2-court-transcription-final")

if __name__ == "__main__":
    main()
