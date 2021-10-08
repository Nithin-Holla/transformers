from datasets import load_from_disk
import re

chars_to_ignore_regex = r'[\_\.\?\,\*\!\:\@]'


def process_text(batch):
    text = batch['text'].lower()
    text = re.sub('-', ' ', text)
    text = re.sub(chars_to_ignore_regex, '', text)
    text = re.sub('\s{2,}', ' ', text)
    batch['text'] = text + " "
    return batch


def get_all_chars(batch):
    all_text = " ".join(batch['text'])
    vocab = list(set(all_text))
    return {"vocab": [vocab],
            "all_text": [all_text]}


def get_vocab_list(data):
    vocab = data.map(get_all_chars,
                     batched=True,
                     batch_size=-1,
                     keep_in_memory=True,
                     remove_columns=data['train'].column_names)

    vocab_list = list(set(vocab['train']["vocab"][0])
                      | set(vocab['validation']["vocab"][0])
                      | set(vocab['test']["vocab"][0]))

    return vocab_list


if __name__ == "__main__":

    ami = load_from_disk("/home/nithinholla/ami_single_headset_segmented_and_chunked")

    ami_text = ami.remove_columns('audio')
    initial_vocab_list = get_vocab_list(ami_text)

    print("Initial vocab list: {}".format(initial_vocab_list))

    ami_processed = ami_text.map(process_text)

    processed_vocab_list = get_vocab_list(ami_processed)

    print("Vocabulary after processing: {}".format(processed_vocab_list))

    vocab_dict = {v: k for k, v in enumerate(processed_vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["<unk>"] = len(vocab_dict)
    vocab_dict["<pad>"] = len(vocab_dict)

    print(vocab_dict)
