gpt2_size = "gpt2"
dataset = "okvqa"
stage = 2
keep = 10
image = "clip"
knowledge_source = "answer"
name = f"{dataset}/path+ans+I_linear/path_{knowledge_source}+I/path1_{keep}_stage{stage}_{gpt2_size}"
args = {
    "info": "",
    "dataset": dataset,
    "batch_size": 30,
    "ckpt": f"./ckpt/{name}/ckpt",
    "train_log_path": f"./ckpt/{name}/train_log.txt",
    "val_log_path": f"./ckpt/{name}/val_log.txt",
    "epochs": 100,
    "stage": 20,
    "num_worker": 16,
    "seed": 35,
    "test_every": 1,
    "learning_rate": 1e-5,
    "path": [1],
    "cuda": "0",
    "keep": keep,
    "args": 0.3,
    "gpt2-size": gpt2_size,
    "model_stage": stage,
    "image": image,
    "ans": False,
    "knowledge_source": knowledge_source
}
print(args)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = args["cuda"]
all_scores = []
from data import get_knowledge_tokenV2

from transformers import AdamW, get_linear_schedule_with_warmup
# from model import KBVQGModel
from predict import generate_beamV2
from eval_tool.eval_tools import Evaluator

import sys

sys.path.append("./")
sys.path.append("./knowledge_path_selecting_model/")
from transformers import GPT2Tokenizer

from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm

DATA_PATH = "xxxx/conceptnet_data"
CLIP_THRESHOLD = 0.9

import torch
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import GPT2LMHeadModel
from transformer import TransformerEncoder
from TransformerMapper import TransformerMapper, TransformerMapperReverse

from data import get_img_clip_embedding, get_ans_clip_embedding, get_answer_tokens, \
    get_question_token_and_mask
from module_utils.util import *


class KBVQGDataset(Dataset):
    def __init__(self, split_type="train", prefix_length=20, args=None):
        super(KBVQGDataset, self).__init__()

        print(f"load {split_type} data")
        self.keep = args["keep"]
        self.paths = args["path"]
        self.questions = pickle.load(open(f"./data/{dataset}/{split_type}.pkl", "rb"))[0]
        self.answers = pickle.load(open(f"./data/{dataset}/{split_type}_most_ans.pkl", "rb"))
        self.imgqids = list(self.questions.keys())
        self.questions = list(self.questions.values())
        self.ans_embed, self.ans_list = get_ans_clip_embedding(split_type=split_type, dataset=dataset)

        self.answer_tokens, self.max_answer_len = get_answer_tokens(split_type=split_type, dataset=dataset)

        self.path_kg, self.max_path_len = get_knowledge_tokenV2(split_type=split_type, path_index=args["path"],
                                                                sentence_type="", source=args["knowledge_source"],
                                                                dataset=dataset)

        if args["image"] == "glip":
            with open(f"./data/glip_object/{dataset}_{split_type}_obj_embed.pkl", "rb") as f:
                self.img_glip_embed = pickle.load(f)
        self.img_embed = get_img_clip_embedding(split_type, dataset)
        self.questions_tokens, self.max_question_len = get_question_token_and_mask(split_type=split_type,
                                                                                   dataset=dataset)
        self.prompt_token = "Question: "
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_size)
        self.gpt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.question_prompt_tokens = torch.tensor(self.gpt_tokenizer.encode(self.prompt_token), dtype=torch.int64)

        self.loss1_prompt = ["question: ",
                             ", reason: ",
                             ". The answer is"]
        self.loss1_prompt_token = [torch.tensor(self.gpt_tokenizer.encode(_), dtype=torch.int64) for _ in
                                   self.loss1_prompt]

        self.prefix_length = prefix_length
        self.split_type = split_type

    def pad_token(self, tokens, length):
        padding = length - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat([tokens, torch.zeros(padding, dtype=torch.int64) - 1])
        elif padding < 0:
            tokens = tokens[:length]

        return tokens

    def pad_tokens_question(self, item: int):
        tokens = self.questions_tokens[item]
        tokens = self.pad_token(tokens, self.max_question_len)
        self.questions_tokens[item] = tokens

        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        return tokens, mask

    def pad_tokens_knowledge(self, idx, item):
        path = self.path_kg[idx - 1][item]
        pad = self.keep - len(path["weight"])

        if len(path["path"]) == 0:
            path["path"] = torch.zeros([pad, self.max_path_len], dtype=torch.int64)
            path["weight"] = torch.zeros(pad, dtype=torch.float32)
            mask = torch.zeros([pad, self.max_path_len], dtype=torch.float32)
            self.path_kg[idx - 1][item] = path
            return path["path"], mask, path["weight"]

        if pad < 0:
            path["path"] = path["path"][:self.keep]
            path["weight"] = path["weight"][:self.keep]

        path["path"] = [self.pad_token(_, self.max_path_len).unsqueeze(0) for _ in path["path"]]
        path["path"] = torch.cat(path["path"], dim=0)
        mask = path["path"].ge(0)
        path["path"][~mask] = 0
        mask = mask.float()

        if pad > 0:
            path_padding = torch.zeros([pad, self.max_path_len], dtype=torch.int64)
            weight_padding = torch.zeros(pad, dtype=torch.float32)
            mask_padding = torch.zeros([pad, self.max_path_len], dtype=torch.float32)
            path["path"] = torch.cat([path["path"], path_padding])
            path["weight"] = torch.cat([path["weight"], weight_padding])
            mask = torch.cat([mask, mask_padding])

        self.path_kg[idx - 1][item] = path
        return path["path"], mask, path["weight"]

    def pad_tokens_answer(self, item):
        tokens = self.answer_tokens[item]
        tokens = self.pad_token(tokens, self.max_answer_len)
        self.answer_tokens[item] = tokens

        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        return tokens, mask

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        question_token, question_mask = self.pad_tokens_question(item)
        answer_token, answer_mask = self.pad_tokens_answer(item)
        ans = self.ans_embed[item]
        # img = self.img_embed[item]
        imgqid = self.imgqids[item]
        if args["image"] == "glip":
            imgid = imgqid.split("#")[0]
            try:
                img = self.img_glip_embed[imgid].sum(0).unsqueeze(0)
            except:
                img = self.img_embed[item]
        else:
            img = self.img_embed[item]
        if len(args["path"]) == 0:
            return img, ans, answer_token, answer_mask, question_token, question_mask
        else:
            paths = []
            paths_mask = []
            paths_weight = []
            if args["knowledge_source"] == "answer":
                for idx in args["path"]:
                    path, path_mask, path_weight = self.pad_tokens_knowledge(idx, item)
                    paths.append(path)
                    paths_mask.append(path_mask)
                    paths_weight.append(path_weight)
            else:
                path, path_mask, path_weight = self.pad_tokens_knowledge(1, item)
                paths.append(path)
                paths_mask.append(path_mask)
                paths_weight.append(path_weight)
            paths = torch.cat(paths)
            paths_mask = torch.cat(paths_mask)
            paths_weight = torch.cat(paths_weight)
            return img, ans, answer_token, answer_mask, paths, paths_mask, paths_weight, question_token, question_mask


def train(train_dataset, val_dataset, model, args):
    max_scores = {}
    model.stage = args["model_stage"]

    val(val_dataset, model, args, -1, max_scores)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        drop_last=True, num_workers=args["num_worker"])
    optimizer = AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=args["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=5000, num_training_steps=args["epochs"] * len(dataloader)
    )
    model = model.cuda()
    to_log(args, "==================================\r")
    for epoch in range(args["epochs"]):
        # if epoch >= args["stage"]:
        #     model.stage = 3
        to_log(args, "=================\r")
        loss_ = 0
        model = model.train()
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            model.zero_grad()
            if len(args["path"]) == 0:
                img, ans, answer_token, answer_mask, question_token, question_mask = [_.cuda() for _ in batch]
                path, path_mask, path_weight = None, None, None

            else:
                img, ans, answer_token, answer_mask, path, path_mask, path_weight, question_token, question_mask = \
                    [_.cuda() for _ in batch]
            gpt_out, ans_loss1, ans_loss2 = model(img=img, ans=ans, answer_token=answer_token, answer_mask=answer_mask,
                                                  prompt_tokens=train_dataset.question_prompt_tokens.cuda(),
                                                  question_token=question_token, question_mask=question_mask, path=path,
                                                  path_mask=path_mask, path_weight=path_weight,
                                                  loss1_prompt_token=[_.cuda() for _ in
                                                                      train_dataset.loss1_prompt_token])
            loss = model.loss_func(ans_loss1, ans_loss2, answer_token, gpt_out, question_token, args["args"])
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_ += loss.item()
        loss_str = f"Loss: {loss_ / len(dataloader)}"
        to_log(args, f"epoch: {epoch}  loss: {loss_str}\r")
        print(loss_str)
        if (epoch + 1) > 10:
            val(val_dataset, model, args, epoch, max_scores)
            model.train()
            # torch.save(
            #     model.state_dict(),
            #     os.path.join(args["ckpt"], f"stage{model.stage}_{epoch:03d}.pt"),
            # )


def val(dataset, model, args, epoch_index, max_scores=None):
    print(f"test epoch {epoch_index}")
    dataloader = DataLoader(
        dataset,
        batch_size=args["batch_size"] * 2 // 3,
        shuffle=False,
        drop_last=False, num_workers=args["num_worker"])
    model.eval()
    generate_texts = []
    model = model.cuda()
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if len(args["path"]) == 0:
            img, ans, answer_token, answer_mask, question_token, question_mask = [_.cuda() for _ in batch]
            path_embed = None
        else:
            img, ans, answer_token, answer_mask, path, path_mask, path_weight, question_token, question_mask = \
                [_.cuda() for _ in batch]
            path_embed = model.knowledge_module(img, ans, path, path_mask, path_weight)

        gpt_out, mask = model.question_module.get_predix_embedding(path=path_embed,
                                                                   prompt_tokens=dataset.question_prompt_tokens.cuda())
        texts = generate_beamV2(model=model.question_module, tokenizer=dataset.gpt_tokenizer,
                                mask=None,
                                entry_length=dataset.max_question_len, embed=gpt_out, batch_size=img.shape[0])
        generate_texts.extend(texts)

    evaluator = Evaluator()
    rtn, scores = evaluator.eval(pred_collect=generate_texts, gt_collect=dataset.questions[:len(generate_texts)],
                                 rtn_num=True)
    to_log(args, f"{epoch_index}\r", mode="val")
    if max_scores != None:
        model_save(scores, max_scores, epoch_index, model, args)

    print(f"======={epoch_index}============")
    for r in rtn:
        to_log(args, r, mode="val")


def gen_result(dataset, model, args, epoch_index, mode="val"):
    knowledge = pickle.load(open(
       f"./knowledge_path_selecting_model/data/okvqa_path_sorted_by_imgembed/{mode}_path1_knowledge_V2_keep100.pkl",
        "rb"))

    print(f"test epoch {epoch_index}")
    dataloader = DataLoader(
        dataset,
        batch_size=args["batch_size"] * 2 // 3,
        shuffle=False,
        drop_last=False, num_workers=args["num_worker"])
    model.eval()
    generate_texts = []
    model = model.cuda()
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if len(args["path"]) == 0:
            img, ans, answer_token, answer_mask, question_token, question_mask = [_.cuda() for _ in batch]
            path_embed = None
        else:
            img, ans, answer_token, answer_mask, path, path_mask, path_weight, question_token, question_mask = \
                [_.cuda() for _ in batch]
            path_embed = model.knowledge_module(img, ans, path, path_mask, path_weight)

        gpt_out, mask = model.question_module.get_predix_embedding(path=path_embed,
                                                                   prompt_tokens=dataset.question_prompt_tokens.cuda())
        texts = generate_beamV2(model=model.question_module, tokenizer=dataset.gpt_tokenizer,
                                mask=None, return_num=5,
                                entry_length=dataset.max_question_len, embed=gpt_out, batch_size=img.shape[0])
        generate_texts.extend(texts)

    result = {}
    for idx, imgqid in tqdm(enumerate(dataset.imgqids)):
        result[imgqid] = {
            "ans": dataset.answers[imgqid],
            "gt": dataset.questions[idx],
            "pr": generate_texts[idx],
            "s": [round(_, 3) for _ in all_scores[idx]],
            "k": knowledge[imgqid]["path"][:dataset.keep]
        }
    import json
    with open(f"./ckpt/{name}/result_{mode}.json", "w") as f:
        json.dump(result, f)


class KnowledgeTestModule(nn.Module):
    def __init__(self):
        super(KnowledgeTestModule, self).__init__()
        self.ans_test_gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, knowledge_tokens, answer_tokens, answer_mask, question_tokens, question_mask, loss1_prompt_token):
        bs = knowledge_tokens.shape[0]
        device = knowledge_tokens.device
        mask_type = question_mask.dtype

        loss1_prompt = [self.ans_test_gpt.transformer.wte(_) for _ in loss1_prompt_token]
        if question_tokens.dtype is not torch.float32:
            feature_question = self.ans_test_gpt.transformer.wte(question_tokens)
        else:
            feature_question = question_tokens
        feature_answer = self.ans_test_gpt.transformer.wte(answer_tokens)
        feature = torch.cat([
            loss1_prompt[0].unsqueeze(0).repeat(bs, 1, 1),
            feature_question,
            loss1_prompt[1].unsqueeze(0).repeat(bs, 1, 1),
            knowledge_tokens,
            loss1_prompt[2].unsqueeze(0).repeat(bs, 1, 1),
            feature_answer
        ], dim=1)

        mask = torch.cat([
            torch.ones(bs, loss1_prompt[0].shape[0], device=device, dtype=mask_type),
            question_mask,
            torch.ones(bs, loss1_prompt[1].shape[0], device=device, dtype=mask_type),
            torch.ones(knowledge_tokens.shape[:2], device=device, dtype=mask_type),
            torch.ones(bs, loss1_prompt[2].shape[0], device=device, dtype=mask_type),
            answer_mask
        ], dim=-1)

        out = self.ans_test_gpt(inputs_embeds=feature, attention_mask=mask)
        return out

    def loss_func(self, out, answer_tokens, in_bs=False, reduction="none"):
        out = out.logits
        out = out[:, -1 * answer_tokens.shape[1] - 1:-1]
        if in_bs is False:
            loss = nnf.cross_entropy(out.reshape(-1, out.shape[-1]), answer_tokens.flatten(), ignore_index=0)
        else:
            loss = nn.CrossEntropyLoss(reduction=reduction)(out.reshape(-1, out.shape[-1]), answer_tokens.flatten())

        return loss


class KnowledgePathModule(nn.Module):
    def __init__(self, src_len):
        super(KnowledgePathModule, self).__init__()
        # 鏌ョ湅clip鍙傛暟

        self.path_score_transformer = TransformerEncoder(512, 4, 8)
        self.path_score_project1 = nn.Linear(512, 1, False)
        self.path_score_project2 = nn.Linear(2, 1, True)

        self.question_predict_transformer = TransformerEncoder(512, 4, 8)
        self.question_predict_ln = nn.Linear(1024, 512, bias=False)
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_size)

        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        if src_len != -1:
            self.path2clip_mapper = TransformerMapperReverse(self.gpt_embedding_size, 512, src_len, src_len,
                                                             num_layers=8)
            self.clip2path_mapper = TransformerMapper(512, self.gpt_embedding_size, src_len, src_len, num_layers=8)

        self.prefix_length = 10
        count = 1
        if args["image"] is not False:
            count += 1
        if args["ans"] is not False:
            count += 1
        self.aggregator = nn.Linear(count * 512, 512, bias=False)

    def _encode_path(self, path_tokens, path_mask):
        out = self.gpt.transformer.wte(path_tokens)
        out = out.reshape(-1, *out.shape[2:])
        out = self.gpt(inputs_embeds=out, attention_mask=path_mask.reshape(-1, path_mask.shape[2]),
                       output_hidden_states=True).hidden_states[-1]
        out = self.path2clip_mapper(out)
        out = out.reshape(*path_mask.shape[:2], out.shape[-1])
        return out

    def _question_predict(self, img, ans):
        ans = ans.unsqueeze(1)
        embed = torch.cat([img, ans], dim=-2).float()
        embed = self.question_predict_transformer(embed.transpose(0, 1)).transpose(0, 1)
        embed = embed / embed.norm(dim=-1, keepdim=True)
        embed = self.question_predict_ln(embed.reshape(embed.shape[0], -1))
        embed = embed / embed.norm(dim=-1, keepdim=True)
        return embed

    def _get_path_score(self, feature, path_embeds, path_weights):
        # 鎰熻杩欎釜鍙互鏀规敼锛屽彧鐢╥mg鍜宲ath鍋歵ransformer
        score_feature = feature.unsqueeze(1).expand(path_embeds.shape)
        score_feature = self.path_score_transformer(score_feature, score_feature, path_embeds)
        score_feature = self.path_score_project1(score_feature)
        score_feature = self.path_score_project2(torch.cat([score_feature, path_weights.unsqueeze(-1)], dim=-1))
        score_feature = nn.functional.softmax(score_feature, dim=-2)
        all_scores.extend(np.round(score_feature.squeeze(2).detach().cpu().numpy(), 3).tolist())
        return score_feature

    # def _cal_ans(self, ans):
    #     feature = None
    #     for a, a_data in ans.items():
    #         if feature is None:
    #             feature = a_data["freq"] * a_data["embed"]
    #         else:
    #             feature = feature + a_data["freq"] * a_data["embed"]
    #     feature = feature / feature.norm(dim=1, keepdim=True)
    #     return feature

    def _get_paths_embed_with_score(self, feature, path_embed=None, path_weight=None):
        feature = self._get_path_score(feature, path_embed, path_weight)
        feature = (path_embed * feature).sum(-2)
        # feature = path_embed.sum(-2)
        feature = feature / feature.norm(dim=1, keepdim=True)
        return feature

    def forward(self, img, ans, path=None, path_mask=None, path_weight=None):
        # feature = self._cal_ans(ans)
        feature = self._question_predict(img, ans)
        path_embed = self._encode_path(path, path_mask)

        path_embed = self._get_paths_embed_with_score(feature, path_embed, path_weight)

        feature = [path_embed]
        if args["image"] is not False:
            feature.append(img.squeeze(1).float())
            # if len(feature) != 1:
            feature = self.aggregator(torch.cat(feature, dim=-1))
        else:
            feature = path_embed

        path_embed = self.clip2path_mapper(feature)
        return path_embed


class QuestionGenerationModule(nn.Module):
    def __init__(self, src_len):
        super(QuestionGenerationModule, self).__init__()
        # self.img_mapper = TransformerEncoder(embed_dim=512, num_heads=4, layers=2, attn_dropout=0, relu_dropout=0,
        #                                      res_dropout=0, embed_dropout=0, attn_mask=True, fc_hid_coeff=4)
        # self.ans_mapper = TransformerEncoder(embed_dim=512, num_heads=4, layers=2, attn_dropout=0, relu_dropout=0,
        #                                      res_dropout=0, embed_dropout=0, attn_mask=True, fc_hid_coeff=4)
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_size)

        self.src_len = src_len

    def get_predix_embedding(self, path, prompt_tokens):
        prompt_tokens = self.gpt.transformer.wte(prompt_tokens).unsqueeze(0).repeat(path.shape[0], 1, 1)
        feature = torch.cat([path, prompt_tokens], dim=1)
        mask = torch.cat(
            [torch.ones(path.shape[:2], device=path.device),
             torch.ones(prompt_tokens.shape[:2], device=path.device)], dim=-1)

        return feature, mask

    def forward(self, path, prompt_tokens, question_token, question_mask):
        feature, mask = self.get_predix_embedding(path, prompt_tokens)
        question_token = self.gpt.transformer.wte(question_token)
        out = self.gpt(inputs_embeds=torch.cat([feature, question_token], dim=1),
                       attention_mask=torch.cat([mask, question_mask], dim=-1), output_hidden_states=True)
        return out

    def loss_func(self, outputs, gt_output):
        logits = outputs.logits[:, -1 * gt_output.shape[1] - 1:-1]
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt_output.flatten(), ignore_index=0)
        return loss


class KBVQGModel(nn.Module):
    def __init__(self, src_len, ):
        super(KBVQGModel, self).__init__()
        print("model initializing")
        self.src_len = src_len
        if self.src_len != -1:
            self.knowledge_module = KnowledgePathModule(src_len)
            # self.knowledge_test = KnowledgeTestModule()
        self.question_module = QuestionGenerationModule(src_len)
        self.stage = 0

    def forward(self, img, ans, answer_token, answer_mask, prompt_tokens, question_token, question_mask,
                path=None, path_mask=None, path_weight=None, loss1_prompt_token=None):
        ans_loss1 = None
        ans_loss2 = None
        if self.src_len != -1:
            path = self.knowledge_module(img, ans, path, path_mask, path_weight)

            out = self.question_module(path, prompt_tokens, question_token, question_mask)
            #
            # ans_loss1 = self.knowledge_test(path, answer_token, answer_mask, question_token, question_mask,
            #                                 loss1_prompt_token)
            #
            # ans_loss2 = self.knowledge_test(path, answer_token, answer_mask,
            #                                 out.hidden_states[-1][:, -1 * question_token.shape[1] - 1:-1],
            #                                 question_mask,
            #                                 loss1_prompt_token)

        else:
            out = self.question_module(path, prompt_tokens, question_token, question_mask)

        return out, ans_loss1, ans_loss2

    def loss_func(self, ans_loss1, ans_loss2, answer_tokens, outputs, gt_output, alpha):
        loss0 = self.question_module.loss_func(outputs, gt_output)
        if self.src_len == -1 or self.stage == 1 or self.stage == 2:
            return loss0

        if self.src_len != -1:
            loss_ans1 = self.knowledge_test.loss_func(ans_loss1, answer_tokens)
            loss_ans2 = self.knowledge_test.loss_func(ans_loss2, answer_tokens, in_bs=True, reduction="mean")
            if self.stage == 0:
                return loss_ans1
            elif self.stage == 3:
                return (1 - alpha) * loss0 + alpha * loss_ans1
            elif model.stage == 4:
                return loss_ans2
            elif model.stage == 5:
                return loss0, loss_ans2
            elif self.stage == 6:
                return alpha * loss_ans1 + (1 - alpha) * loss0, loss_ans2
            elif self.stage == 7:
                return alpha / 2 * (loss_ans1 + loss_ans2) + (1 - alpha) * loss0
            elif self.stage == 8:
                return alpha * loss_ans2 + (1 - alpha) * (alpha * loss_ans1 + (1 - alpha) * loss0)
            elif self.stage == 9:
                return (1 - alpha) * loss0 + alpha * loss_ans2

    def train(self, mode=True):
        if self.stage == 0:
            if self.src_len != -1:
                self.knowledge_module.train(mode)
                self.knowledge_test.train(False)
            self.question_module.train(False)

        elif self.stage == 1:
            if self.src_len != -1:
                self.knowledge_module.train(False)
                self.knowledge_test.train(False)
            self.question_module.train(mode)

        else:
            if self.src_len != -1:
                self.knowledge_module.train(mode)
                # self.knowledge_test.train(False)
            self.question_module.train(mode)

        return self

    def eval(self):
        self.train(False)
        return self


if __name__ == '__main__':
    seed_torch(args["seed"])
    os.makedirs(args["ckpt"], exist_ok=True)
    to_log(args, str(args))
    val_dataset = KBVQGDataset("val", args=args)
    train_dataset = KBVQGDataset("train", args=args)
    model = KBVQGModel(train_dataset.max_path_len)
    print("start training")
    train(train_dataset, val_dataset, model, args)

    # val(val_dataset, model, args, i)
    # model.load_state_dict(torch.load(""))
    # gen_result(val_dataset, model, args, 30, "val")
