import torch
from torch.utils.data import DataLoader
from data_loader import load_user_posts, load_user_metaphors
from typing import Iterator, List, Dict, Union, Tuple, Optional
import logging
import datetime
from datetime import datetime
from pprint import pprint as pp
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch.optim as optim
import sys
import operator
import os
from transformers import AutoTokenizer, AutoModel
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding, FeedForward, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util, InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn.util import get_text_field_mask
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers import MultiTaskDatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, PretrainedTransformerTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.common import Params
from allennlp.predictors import Predictor
from allennlp.common import JsonDict
from allennlp.training.trainer import Trainer
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.nn import Activation, InitializerApplicator
from sentence_transformers import SentenceTransformer
from overrides import overrides
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

MAXIMUM_POST_SEQ_SIZE = 100
ATTENTION_OPTION_NONE = 0  # no attention
ATTENTION_OPTION_ATTENTION_WITH_POST = 1
ATTENTION_OPTION_ATTENTION_WITH_METAPHOR = 1
POST_ENCODER_OPTION_LSTM = 1
METAPHOR_ENCODER_OPTION_LSTM = 1

print("CUDA AVAILABILITY: {}".format(torch.cuda.is_available()))

def timestamped_print(msg):
    logger.info(msg)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(ts + " :: " + msg)


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



@DatasetReader.register("depression_data_reader")
class DepressionDataReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        super().__init__()
        timestamped_print("DepressionDataReader ...")
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}


    @overrides
    def text_to_instance(self, user_id: str, tag: str=None) -> Instance:
        fields: Dict[str, Field] = {}
        fields['user_id'] = MetadataField(str(user_id))
        if tag is not None:
            fields['label'] = LabelField(int(tag), skip_indexing=True)
        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        print(file_path)

        df = pd.read_csv(file_path, header=0, encoding='utf-8', delimiter=',', lineterminator='\n', \
                         usecols=range(0,2)).values
        # df = pd.read_csv(file_path)
        for data_row in df[:]:
            # label = 1 if data_row[0] == 'depression' else 0
            user_id = data_row[1]
            label = data_row[0]
            yield self.text_to_instance(user_id=user_id, tag=label)

    
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, hidden_dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(hidden_dim)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, key)
        return context, attn

    
class HAN_block(nn.Module):
    '''
    Attention mechanism is one of the above attentions.
    Args:
        hidden_dim: dimesion of embedding matrix
    Inputs: query, key
         - **query** (batch_size, 1, hidden_dim): for the first layer, it is a trainable randomly initialized vectors in a batch; for non-first layer it is the context output of previous layer.
         - **key** (batch_size, max_len, hidden_dim): embedding matrix.
    '''
    def __init__(self, hidden_dim):
        super(HAN_block, self).__init__()
        self.att = ScaledDotProductAttention(hidden_dim)
        self.linear_observer = nn.Linear(hidden_dim,hidden_dim)
        self.linear_matrix = nn.Linear(hidden_dim,hidden_dim)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.2) # need to modify the dropout rate accordingly
        
    def forward(self, query: Tensor, key: Tensor):
        query_ = query[:key.size(0),:,:] # make sure that the batch size of query matches the size of key in the last batch of an epoech
        context, att_weight = self.att(query_,key)
        new_query_vec = self.dropout(self.layer_norm(self.activation(self.linear_observer(context))))
        new_key_matrix = self.dropout(self.layer_norm(self.activation(self.linear_matrix(key))))
        
        return new_query_vec, new_key_matrix, att_weight


@Model.register("depression_classifier")
class DepressionClassifier(Model):
    def __init__(self,  
                 vocab: Vocabulary,
                 post_encoder: Seq2SeqEncoder,
                 metaphor_encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 cuda_device: int = -1,
                 max_post_size: int=MAXIMUM_POST_SEQ_SIZE,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ) -> None:

        super().__init__(vocab)
        self.num_tags = 0
        if self.vocab is not None:
            self.num_tags = self.vocab.get_vocab_size('label')

        self.post_encoder = post_encoder
        self.metaphor_encoder = metaphor_encoder

        self.classifier_feedforward = classifier_feedforward
        self.cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
        print("CURRENT CUDA DEVICE ", self.cuda_device)
        self.max_post_size = max_post_size


        LABEL_TYPE_DEPRESSION: int = 1
        self.accuracy = CategoricalAccuracy()
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1": F1Measure(positive_label=LABEL_TYPE_DEPRESSION)
        }
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.cuda_device) \
            if torch.cuda.is_available() else torch.nn.CrossEntropyLoss()

        ### Loading local files
        self.embedding_tokenizer = AutoTokenizer.from_pretrained('[your_dir]/bert-base-uncased')
        self.embedding_model = AutoModel.from_pretrained('[your_dir]/bert-base-uncased').to(self.cuda_device)

        self.tweet_query0 = None
        self.meta_query0 = None
        self.bn_input = None
        initializer(self)

        self.HAN_1_meta = HAN_block(768)
        self.HAN_2_meta = HAN_block(768)
        self.HAN_1_tweet = HAN_block(768)
        self.HAN_2_tweet = HAN_block(768)


    def set_max_post_size(self, max_post_size: int = MAXIMUM_POST_SEQ_SIZE):
        if max_post_size:
            self.max_post_size = max_post_size
            timestamped_print("maximum post sequence size is [%s]" % self.max_post_size)
            timestamped_print("re-set attention1 with the new post sequence size")


    def forward(
            self,
            # text: Dict[str, torch.Tensor],
            user_id: List,
            label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:


        content_tensor_in_batch_padded, batch_content_mask, metaphor_tensor_in_batch_padded, batch_metaphor_mask\
            = self.batch_encoding(user_id, maximum_sequence_length=self.max_post_size)

        timestamped_print("encode social context with LSTM")

        if self.tweet_query0 == None:
            self.tweet_query0 = torch.rand([content_tensor_in_batch_padded.size(0), 1, content_tensor_in_batch_padded.size(2)], requires_grad = True)
        tweet_query1, tweet_key1, att_weight_1 = self.HAN_1_tweet(self.tweet_query0, content_tensor_in_batch_padded)
        tweet_query2, tweet_key2, att_weight_2 = self.HAN_2_tweet(tweet_query1, tweet_key1)
        post_encoder_out = tweet_query2.squeeze(1)

        timestamped_print("post reprsentation shape : %s" % (str(post_encoder_out.shape)))

        if self.meta_query0 == None:
            self.meta_query0 = torch.rand([metaphor_tensor_in_batch_padded.size(0), 1, metaphor_tensor_in_batch_padded.size(2)], requires_grad = True)
        meta_query1, meta_key1, att_weight_1 = self.HAN_1_meta(self.meta_query0, metaphor_tensor_in_batch_padded)
        meta_query2, meta_key2, att_weight_2 = self.HAN_2_meta(meta_query1, meta_key1)
        metaphor_encoder_out = meta_query2.squeeze(1)

        timestamped_print("metaphor reprsentation shape : %s" % (str(metaphor_encoder_out.shape)))

        # final_representation = post_encoder_out
        final_representation = torch.cat((post_encoder_out, metaphor_encoder_out), dim=1)
        timestamped_print("user's posts representation before feedforward: %s" % str(final_representation.shape))
        logits = self.classifier_feedforward(final_representation)
        output_dict = {"logits": logits}
        if label is not None:
            loss = self.loss_function(logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_measures = self.metrics["f1"].get_metric(reset=reset)
        print(f1_measures)
        print()
        return {
            # https://github.com/allenai/allennlp/issues/1863
            # f1 get_metric returns (precision, recall, f1)
            "precision": f1_measures['precision'],
            "recall": f1_measures['recall'],
            "f1": f1_measures['f1'],
            "accuracy": self.metrics["accuracy"].get_metric(reset=reset)
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DepressionClassifier':  # type: ignore

        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))

        post_encoder = Seq2SeqEncoder.from_params(params.pop("post_encoder"))
        metaphor_encoder = Seq2SeqEncoder.from_params(params.pop("metaphor_encoder"))


        initializer = InitializerApplicator.from_params(params.pop('initializer', []))

        return cls(
                   vocab=vocab,
                   classifier_feedforward=classifier_feedforward,
                   post_encoder=post_encoder,
                   metaphor_encoder=metaphor_encoder,
                   initializer=initializer)


    def batch_encoding(self, user_ids: List, maximum_sequence_length=MAXIMUM_POST_SEQ_SIZE):
        # print()
        # pp(user_ids)
        timestamped_print("encode posts per user in batch ...")
        timestamped_print("First two user ids in current batch: [%s] and [%s]" % (str(user_ids[0]),
                                                                                  str(user_ids[1])))


        post_set_list = [list(load_user_posts(user_id)) for user_id in user_ids]
        post_set_list = [post_list[:maximum_sequence_length] for post_list in post_set_list]

        content_tensor_in_batch = []

        for user_id, individual_post_set in zip(user_ids, post_set_list):
            content_tensor = self._context_sequence_encoding(user_id, individual_post_set, content_option='post')
            content_tensor_in_batch.append(content_tensor)

        metaphor_set_list = []
        for user_id in user_ids:
            metaphor_set_list.append(list(load_user_metaphors(user_id))[:maximum_sequence_length])
        metaphor_tensor_in_batch = []
        for user_id, individual_metaphor_set in zip(user_ids, metaphor_set_list):
            metaphor_tensor = self._context_sequence_encoding(user_id, individual_metaphor_set, content_option='metaphor')
            metaphor_tensor_in_batch.append(metaphor_tensor)


        if len(content_tensor_in_batch) != len(user_ids):
            # check/debug
            timestamped_print(
                "Error: context propagation encoding size [%s] does not match the size of input posts [%s]. "
                % (len(content_tensor_in_batch), len(user_ids)))

        if len(metaphor_tensor_in_batch) != len(user_ids):
            # check/debug
            timestamped_print(
                "Error: context propagation encoding size [%s] does not match the size of input posts [%s]. "
                % (len(metaphor_tensor_in_batch), len(user_ids)))


        timestamped_print(
            "done post & metaphor encoding! Padding and normalise sequence tensors for current batch now...")

        content_tensor_in_batch_padded = torch.Tensor()
        batch_content_mask = torch.Tensor()

        content_tensor_in_batch_padded, batch_content_mask = self.padding_and_norm_propagation_tensors(
            content_tensor_in_batch,
            maximum_sequence_length=maximum_sequence_length)

        metaphor_tensor_in_batch_padded = torch.Tensor()
        batch_metaphor_mask = torch.Tensor()

        metaphor_tensor_in_batch_padded, batch_metaphor_mask = self.padding_and_norm_propagation_tensors(
            metaphor_tensor_in_batch,
            maximum_sequence_length=maximum_sequence_length)

        timestamped_print(
            "Done. propagation tensors after batch normalisation: post content shape [%s], metaphor content shape [%s]" %
            (str(content_tensor_in_batch_padded.size()), str(metaphor_tensor_in_batch_padded.size())))

        timestamped_print("post masking: content [%s]" % (str(batch_content_mask.shape)))

        return content_tensor_in_batch_padded, batch_content_mask, metaphor_tensor_in_batch_padded, batch_metaphor_mask


    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalise predictions.
        This method overrides `Model.make_output_human_readable`, which gets called after `Model.forward`, at test
        time, to finalise predictions. The logic for the decoder part of the encoder-decoder lives
        within the `forward` method.
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called `predicted_tokens` to the `output_dict`.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=2)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="label")
                  for x in argmax_indices]
        timestamped_print("labels <- decode: %s" % labels)

        output_dict['label'] = labels
        return output_dict


    def padding_and_norm_propagation_tensors(self, list_of_context_seq_tensor: List[torch.FloatTensor],
                                             maximum_sequence_length=100) -> (torch.FloatTensor, torch.uint8):
        """
        padding and masking

        :param list_of_context_seq_tensor:
        :param maximum_cxt_sequence_length:
        :return: padded batch context tensor, batch context mask (for attention)
        """
        try:
            batch_cxt_mask = self.creating_batch_context_tensor_mask(list_of_context_seq_tensor,
                                                                     maximum_sequence_length)

            feature_dim = list_of_context_seq_tensor[0].shape[1]
            list_of_context_seq_tensor.insert(0, torch.zeros(maximum_sequence_length, feature_dim))
            # zero padding
            batch_propagation_tensors = torch.nn.utils.rnn.pad_sequence(list_of_context_seq_tensor, batch_first=True)
            # remove the dummy tensor
            batch_propagation_tensors = batch_propagation_tensors[1:]

        except RuntimeError as err:
            print(err)
            timestamped_print(
                "error when pad social-context feature tensors. Check all tensor shapes as following in current batch ...")
            for context_seq_tensor in list_of_context_seq_tensor:
                timestamped_print(str(context_seq_tensor.shape))
            timestamped_print("done")

            raise err

        timestamped_print("tensor size after padding: %s" % str(
            batch_propagation_tensors.size()))  # -> (batch_size, padded size of context sequence, dimension of instance reqpresentation)

        batch_size = batch_propagation_tensors.size()[0]
        if self.bn_input:
            normalised_propagation_tensor = torch.stack(
                [self.bn_input(batch_propagation_tensors[i]) for i in range(batch_size)])
        else:
            normalised_propagation_tensor = torch.stack([batch_propagation_tensors[i] for i in range(batch_size)])

        return normalised_propagation_tensor.float(), batch_cxt_mask


    def creating_batch_context_tensor_mask(self, list_of_context_seq_tensor: List[torch.FloatTensor],
                                           maximum_sequence_length=300):
        """
        creating mask for attention
        :param context_tensor_seq:
        :return: ByteTensor (a Boolean tensor) (torch.uint8)
        """
        vary_cxt_lengths = torch.stack(
            [torch.as_tensor(tensor_cxt_tensor.shape[0], dtype=torch.float) for tensor_cxt_tensor in
             list_of_context_seq_tensor])

        idxes = torch.arange(0, maximum_sequence_length,
                             out=torch.FloatTensor(maximum_sequence_length)).unsqueeze(0)


        if torch.cuda.is_available():
            idxes = idxes.cuda()
            vary_cxt_lengths = vary_cxt_lengths.cuda()
        else:
            idxes.cpu()
            vary_cxt_lengths.cpu()

        mask = idxes < vary_cxt_lengths.unsqueeze(1)

        return mask


    def _context_sequence_encoding(self, user_id: str, individual_post_set: List[Dict], content_option) -> (
            torch.FloatTensor, torch.FloatTensor):
        """
        prepare sorted post sequence:
        :param: user ids
        :param: individual_post_set: posts per user
        :return:List[torch.FloatTensor], sequence tensor from temporally sorted encodings of posts
        """
        EXPECTED_ENCODER_INPUT_DIM = 768
        if content_option =='post' and self.post_encoder:
            EXPECTED_ENCODER_INPUT_DIM = self.post_encoder.get_input_dim()
        elif content_option=='metaphor' and self.metaphor_encoder:
            EXPECTED_ENCODER_INPUT_DIM = self.metaphor_encoder.get_input_dim()

        # print("len(individual_post_set) ", len(individual_post_set))
        if len(individual_post_set) == 0:

            post_content_seq_tensor = torch.zeros(1, EXPECTED_ENCODER_INPUT_DIM)
        else:
            try:
                all_post_content_embeddings = []
                for post in individual_post_set:
                    post_embedding = encode_content_manual(self.post_encoder, self.embedding_model, self.embedding_tokenizer, post)
                    all_post_content_embeddings.append(post_embedding)
                post_content_seq_tensor = self.sort_encoding_with_time_sequence(individual_post_set, all_post_content_embeddings)

            except:
                print("Unexpected error when encoding user [%s]'s posts" % user_id, sys.exc_info()[0])
                raise

        return post_content_seq_tensor


    def sort_encoding_with_time_sequence(self, individual_post_set: List[Dict],
                                         all_post_content_embeddings: List[np.ndarray]) -> (
                                        torch.FloatTensor, torch.FloatTensor):
        """
        put post representation in temporal order before feeding into LSTM
        """
        cxt_size = len(individual_post_set)
        content_embedding_time_seq = []
        for i in range(0, cxt_size):
            post_i = individual_post_set[i]
            post_embedding_i = all_post_content_embeddings[i]

            post_timestamp = datetime.strptime(post_i["timestamp"],
                                               '%Y-%m-%d %H:%M:%S')  # timestamp
            content_embedding_time_seq.append((post_timestamp, post_embedding_i))

        content_embedding_time_seq_sorted = sorted(content_embedding_time_seq[:], key=operator.itemgetter(0),
                                                   reverse=False)

        content_seq_representation: List[np.ndarray] = [post_representation[1] for post_representation in
                                                        content_embedding_time_seq_sorted]
        content_seq_representation = np.array(content_seq_representation)
        return torch.as_tensor(content_seq_representation, dtype=torch.float32)


@Predictor.register('depression_user_tagger')
class DepressionUserTaggerPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        # self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    def predict(self, user_id: str) -> JsonDict:
        """

        :param user_id:
        :return: dict, {'logits': [], 'class_probabilities': [], 'label': str}
        """
        print("type(self): ", type(self._model))
        return self.predict_json({"user_id": user_id})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        user_id = json_dict["user_id"]
        return self._dataset_reader.text_to_instance(user_id)


def model_training(train_set_path, validation_set_path, test_set_path, n_gpu: Union[int, List] = -1,
                   train_batch_size: int = 100, model_file_prefix="", num_epochs: int = 2,
                   max_post_size_option: int = MAXIMUM_POST_SEQ_SIZE):
    """
    https://guide.allennlp.org/training-and-prediction#1
    :param train_set_path:
    :param validation_set_path:
    :param test_set_path:
    :param n_gpu:
    :param train_batch_size:
    :param model_file_prefix:
    :param num_epochs:
    :param post_encoder_option:
    :param max_post_size_option:
    :return:
    """
    timestamped_print(
        "start to train ExplainableDepressionDetection model with training set [%s] and dev set [%s] with gpu [%s] ... " % (
            train_set_path, validation_set_path, n_gpu))
    timestamped_print("the model will be evaluated with test set [%s]" % test_set_path)
    # enable GPU here
    n_gpu = config_gpu_use(n_gpu)
    timestamped_print("training batch size: [%s]" % train_batch_size)

    token_indexer = ELMoTokenCharactersIndexer()
    train_reader = DepressionDataReader(token_indexers={'elmo': token_indexer})
    validation_reader = DepressionDataReader(token_indexers={'elmo': token_indexer})
    timestamped_print("loading development dataset and indexing vocabulary  ... ")
    train_set = list(train_reader.read(train_set_path))
    validation_set = list(validation_reader.read(validation_set_path))
    vocab = Vocabulary.from_instances(train_set+validation_set)
    train_loader = MultiProcessDataLoader(train_reader, train_set_path, batch_size=train_batch_size, shuffle=True)
    validation_loader = MultiProcessDataLoader(validation_reader, validation_set_path, batch_size=train_batch_size, shuffle=True)
    train_loader.index_with(vocab)
    validation_loader.index_with(vocab)


    timestamped_print("done. datasets loaded and vocab indexed completely.")

    timestamped_print("initialising ExplainableDepressionDetection model ... ")
    model = instantiate_model(n_gpu, vocab)
    model.set_max_post_size(max_post_size_option)

    timestamped_print("model architecture: ")
    print(model)
    timestamped_print("done. ExplainableDepressionDetection model is initialised completely.")
    timestamped_print("initialising optimiser and dataset iteractor ... ")
    optimiser = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    timestamped_print("done.")

    timestamped_print("training starting now ... ")
    trainer = GradientDescentTrainer(
        model=model,
        optimizer=optimiser,
        data_loader=train_loader,
        validation_data_loader = validation_loader,
        num_epochs=num_epochs,
        cuda_device=n_gpu
    )
    trainer.train()
    timestamped_print("done.")

    try:
        archive_model_from_memory(model, vocab, model_file_prefix)
    except AttributeError as err:
        timestamped_print("failed to archive model ... ")
        print(err)

    # quick_test(model)
    evaluation(test_set_path, model, n_gpu)


def encode_content(content_encoder, embedding_model, post_dict):
    """
    Loading a pre-trained LM from the sentence-transformer library directly
    :param content_encoder:
    :param embedding_model:
    :param post_dict:
    :return:
    """
    post_embedding = np.zeros(content_encoder.get_input_dim())
    post_text = post_dict['text']
    ## TODO: process text (tokenisation)
    # post_tokens = preprocessing_text(post_text)
    if len(post_text) > 0:
        ## TODO: replace embedding model
        post_embedding = embedding_model.encode(post_text)
        # print(type(post_embedding))

    else:
        post_embedding = np.zeros(content_encoder.get_input_dim())

    return post_embedding


def encode_content_manual(content_encoder, embedding_model, embedding_tokenizer, post_dict):
    # print("post in encode_content_manual", post_dict )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    post_embedding = np.zeros(content_encoder.get_input_dim())
    # post_text = post_dict['text']


    ## TODO: process text (tokenisation)
    # post_tokens = preprocessing_text(post_text)
    if 'text' in post_dict:
        post_text = post_dict['text']
        if len(post_text) > 0:
            encoded_input = embedding_tokenizer(post_text, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = embedding_model(**encoded_input)
            post_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
            post_embedding = F.normalize(post_embedding, p=2, dim=1)
            post_embedding = post_embedding[0].cpu().data.numpy()

    else:
        post_embedding = np.zeros(content_encoder.get_input_dim())

    return post_embedding



def config_gpu_use(n_gpu: Union[int, List] = -1) -> Union[int, List]:
    """
    set GPU device

    set to 0 for 1 GPU use

    Notes: 1)Dataloaders give normal (non-cuda) tensors by default. They have to be manually cast using the Tensor.to() method.
    2) Many methods are simply not implemented for torch.cuda.*Tensor. Thus, setting the global tensor type to cuda fails.
    3) Conversions to numpy using the numpy() method aren’t’ available for cuda tensors. One has to go x.cpu().numpy().

    :param n_gpu:
    :return:
    """
    if n_gpu != -1:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    return n_gpu

def concat_generators(*args):
    for gen in args:
        yield from gen



def quick_test(model_in_memory: DepressionClassifier):
    # test the classifier
    timestamped_print("prediction test on trained depression classifier: ")
    try:
        token_indexer = ELMoTokenCharactersIndexer()
        train_reader = DepressionDataReader(token_indexers={'elmo': token_indexer})


        predictor = DepressionUserTaggerPredictor(model_in_memory, dataset_reader=train_reader)
        user_id = "8180040"

        outputs = predictor.predict(user_id)
        print("predictor output: ", outputs)

        print("print vocab: ")
        model_in_memory.vocab.print_statistics()
        # model.vocab.get_token_from_index(label_id, 'labels')
        timestamped_print("prediction label on (%s): %s" % (
            user_id, outputs["label"] if "label" in outputs else "label is unknown"))
    except Exception as e:
        timestamped_print("errors in quick model test ")
        print(e)



def evaluation(test_data_path, model_in_memory: Model, cuda_device=-1):

    timestamped_print("evaluating  .... ")

    token_indexer = ELMoTokenCharactersIndexer()
    test_reader = DepressionDataReader(token_indexers={'elmo': token_indexer})
    test_instances = list(test_reader.read(test_data_path))
    vocab = Vocabulary.from_instances(test_instances)
    test_loader = MultiProcessDataLoader(test_reader, test_data_path, batch_size=32, shuffle=True)
    test_loader.index_with(vocab)

    batch_iterator = iter(test_loader)

    import ntpath
    output_file_prefix = ntpath.basename(test_data_path)
    output_file_prefix = output_file_prefix.replace(".", "_")

    from training_util import evaluate

    metrics = evaluate(model_in_memory, test_instances, batch_iterator, cuda_device, "")
    timestamped_print("Finished evaluating.")
    timestamped_print("Metrics:")
    for key, metric in metrics.items():
        print("%s: %s" % (key, metric))

    output_file = os.path.join(os.path.dirname(__file__), '..',
                               '..', "data", "test",)
    os.makedirs(output_file, exist_ok=True)
    import json
    if output_file:
        with open(os.path.join(output_file, output_file_prefix + "_eval.json"), "w") as file:
            json.dump(metrics, file, indent=4)

    print("completed")



def archive_model_from_memory(model_in_memory: Model, vocab: Vocabulary, file_prefix=""):
    """
    https://allennlp.org/tutorials

    :param model_in_memory:
    :param vocab:
    :param file_prefix, optional model file name prefix
    :return:
    """
    timestamped_print("archive model and vocabulary ... ")
    import time

    model_timestamp_version = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
    serialization_dir = os.path.join(os.path.dirname(__file__), '..', '..', "output", file_prefix + model_timestamp_version)
    os.makedirs(serialization_dir, exist_ok=True)
    vocab_dir_name = "vocabulary"
    weights_file_name = "weights_best.th"

    if not os.path.exists(serialization_dir):
        os.mkdir(serialization_dir)

    vocab_dir_path = os.path.join(serialization_dir, vocab_dir_name)
    weights_file_path = os.path.join(serialization_dir, weights_file_name)

    model_state = model_in_memory.state_dict()
    with open(weights_file_path, 'wb') as f:
        torch.save(model_state, f)

    vocab.save_to_files(vocab_dir_path)

    timestamped_print("done. model file and vocab file are archived in [%s]" % (serialization_dir))




def instantiate_model(n_gpu, vocab, max_post_size: int = MAXIMUM_POST_SEQ_SIZE):
    post_encoder = None
    metaphor_encoder = None
    POST_EMBEDDING_DIM = 768
    POST_LSTM_INPUT_DIM = POST_EMBEDDING_DIM
    POST_LSTM_HIDDEN_DIM = POST_LSTM_INPUT_DIM * 2
    METAPHOR_EMBEDDING_DIM = 768
    METAPHOR_LSTM_INPUT_DIM = POST_EMBEDDING_DIM
    METAPHOR_LSTM_HIDDEN_DIM = POST_LSTM_INPUT_DIM * 2

    post_encoder = torch.nn.LSTM(POST_LSTM_INPUT_DIM,
                                 POST_LSTM_HIDDEN_DIM, num_layers=2, batch_first=True,
                                 bidirectional=False)

    metaphor_encoder = torch.nn.LSTM(METAPHOR_LSTM_INPUT_DIM,
                                     METAPHOR_LSTM_HIDDEN_DIM, num_layers=2, batch_first=True,
                                 bidirectional=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # post_encoder.to(device)

    post_encoder = PytorchSeq2SeqWrapper(post_encoder, stateful=False)
    metaphor_encoder = PytorchSeq2SeqWrapper(metaphor_encoder, stateful=False)

    feedforward_input_dim = post_encoder.get_output_dim() + metaphor_encoder.get_output_dim()
    feedforward_hidden_dim_1 = feedforward_input_dim
    feedforward_hidden_dim_2 = int(feedforward_input_dim / 2)

    class_num = 2
    classifier_feedforward = FeedForward(
                                         input_dim=feedforward_hidden_dim_2, num_layers=3,
                                         hidden_dims=[feedforward_hidden_dim_2, feedforward_hidden_dim_2, class_num],
                                         activations=[Activation.by_name("leaky_relu")(),
                                                      Activation.by_name("linear")(),
                                                      Activation.by_name("linear")()],
                                         dropout=[0.2, 0.3, 0.3])

    timestamped_print("set fully-connected layer to leaky_relu+linear+linear with dropout (0.2, 0.3, 0.0)")

    classifier_feedforward.to(device)


    model = DepressionClassifier(vocab,
                                 post_encoder=post_encoder,
                                 metaphor_encoder=metaphor_encoder,
                                 classifier_feedforward=classifier_feedforward,
                                 max_post_size=int(max_post_size),
                                 cuda_device=n_gpu)
    model.to(device)

    return model


def load_classifier_from_archive(vocab_dir_path: str = None,
                                 model_weight_file: str = None,
                                 n_gpu_use: Union[int, List] = -1,
                                 max_post_size: int = MAXIMUM_POST_SEQ_SIZE,
                                 post_attention_option: int = ATTENTION_OPTION_ATTENTION_WITH_POST,
                                 metaphor_attention_option: int = ATTENTION_OPTION_ATTENTION_WITH_METAPHOR) -> Tuple[
    DepressionClassifier, DepressionUserTaggerPredictor]:

    model_timestamp_version = ""
    serialization_dir = os.path.join(os.path.dirname(__file__), '..', '..', "output", model_timestamp_version)
    vocab_dir_name = "vocabulary"
    weights_file_name = "weights_best.th"

    if model_weight_file is None:
        model_weight_file = os.path.join(serialization_dir, weights_file_name)

    if vocab_dir_path is None:
        vocab_dir_path = os.path.join(serialization_dir, vocab_dir_name)

    n_gpu = config_gpu_use(n_gpu_use)
    vocab = Vocabulary.from_files(vocab_dir_path)
    model = instantiate_model(n_gpu, vocab, max_post_size=max_post_size)

    model.set_max_post_size(max_post_size=max_post_size)

    with open(model_weight_file, 'rb') as f:
        model.load_state_dict(torch.load(f))

    token_indexer = ELMoTokenCharactersIndexer()
    test_reader = DepressionDataReader(token_indexers={'elmo': token_indexer})
    dnn_predictor = DepressionUserTaggerPredictor(model, dataset_reader=test_reader)

    return model, dnn_predictor



if __name__ == '__main__':
    test()

