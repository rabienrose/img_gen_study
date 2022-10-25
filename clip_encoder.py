from torch import optim, nn
import torch
import util
from einops import rearrange, repeat
import numpy as np
from transformers import CLIPTokenizer, CLIPConfig
from transformers import CLIPModel


class FrozenCLIPEmbedder(nn.Module):
    def __init__(self, version="./models/openai--clip-vit-large-patch14", device="cpu", max_length=77):  # clip-vit-base-patch32
        super().__init__()
        self.use_prior = False
        self.return_layer = None # set: model.cond_stage_model.return_layer = -2 # penultimate layer
        self.do_final_ln = False # set: model.cond_stage_model.do_final_ln = True # make it kinda work right away (or don't and let the model figure it out)
        self.inference_mode = True # processes () and []
        self.clip_extend = False
        self.max_clip_extend = 75
        self.emphasis_factor = 1.05 # strength of () and []
        config = CLIPConfig.from_pretrained(version)
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPModel.from_pretrained(version).text_model #CLIPTextModel.from_pretrained(version, config=config.text_config)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        self.freeze()

        self.token_mults = {}
        tokens_with_parens = [(k, v) for k, v in self.tokenizer.get_vocab().items() if '{' in k or '}' in k or '[' in k or ']' in k]
        fac = self.emphasis_factor
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= fac
                if c == ']':
                    mult *= fac
                if c == '{':
                    mult *= fac
                if c == '}':
                    mult /= fac
            if mult != 1.0:
                self.token_mults[ident] = mult

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, input_ids=None, input_multipliers=None):
        assert(self.max_clip_extend % 75 == 0) # only full context sizes supported
        future = None
        if self.inference_mode:
            remade_batch_tokens = []
            batch_multipliers = []

            if input_ids is None:
                batch_tokens = self.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
                # get length of longest batch items, min 75
                longest = 0
                for tokens in batch_tokens:
                    if len(tokens) > longest:
                        longest = len(tokens)
                target_len = max(75, int(75 * np.ceil(longest/75.)))

                # parse brackets
                for tokens in batch_tokens:
                    fixes = []
                    remade_tokens = []
                    multipliers = []
                    mult = 1.0

                    for token in tokens:
                        mult_change = self.token_mults.get(token)
                        if mult_change is not None:
                            mult *= mult_change
                        else:
                            remade_tokens.append(token)
                            multipliers.append(mult)

                    need = (target_len-len(remade_tokens))
                    if need >= 1:
                        remade_tokens.extend([self.tokenizer.eos_token_id] * need)
                        multipliers.extend([1.0] * need)

                    # limit length to maximum extension length
                    remade_tokens = remade_tokens[:self.max_clip_extend]
                    multipliers = multipliers[:self.max_clip_extend]

                    remade_batch_tokens.append(remade_tokens)
                    batch_multipliers.append(multipliers)

                remade_batch_tokens = torch.LongTensor(remade_batch_tokens).to(self.device)
                batch_multipliers = torch.FloatTensor(batch_multipliers).to(self.device)
            else:
                remade_batch_tokens = input_ids
                batch_multipliers = input_multipliers

            # recursively call self again if context too long
            if self.clip_extend and remade_batch_tokens.shape[1] > 75:
                future = self.forward("", input_ids=remade_batch_tokens[:, 75:], input_multipliers=batch_multipliers[:, 75:])

            # cut off contexts with high length
            remade_batch_tokens = remade_batch_tokens[:, :75]
            batch_multipliers = batch_multipliers[:, :75]

            # add special tokens
            ones = torch.ones((remade_batch_tokens.shape[0], 1), dtype=torch.long).to(self.device)
            remade_batch_tokens = torch.cat((ones * self.tokenizer.bos_token_id, remade_batch_tokens, ones * self.tokenizer.eos_token_id), dim=1)
            batch_multipliers = torch.cat((ones, batch_multipliers, ones), dim=1)

            tokens = remade_batch_tokens
        else:
            if not self.clip_extend:
                tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")["input_ids"].to(self.device)
            else:
                if input_ids is None:
                    #tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")["input_ids"].to(self.device)
                    tokens = self.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]

                    # get length of longest batch items, min 75
                    longest = 0
                    for ts in tokens:
                        if len(ts) > longest:
                            longest = len(ts)
                    target_len = max(75, int(75 * np.ceil(longest/75.)))

                    for ts in tokens:
                        need = (target_len-len(ts))
                        if need >= 1:
                            ts.extend([self.tokenizer.eos_token_id] * need)

                    tokens = torch.LongTensor(tokens).to(self.device)
                else:
                    tokens = input_ids

                # limit length to maximum extension length
                tokens = tokens[:, :self.max_clip_extend]

                # recursively call self again if context too long
                if self.clip_extend and tokens.shape[1] > 75:
                    future = self.forward("", input_ids=tokens[:, 75:])

                # cut off contexts with high length
                tokens = tokens[:, :75]

                # add special tokens
                ones = torch.ones((tokens.shape[0], 1), dtype=torch.long).to(self.device)
                tokens = torch.cat((ones * self.tokenizer.bos_token_id, tokens, ones * self.tokenizer.eos_token_id), dim=1)

        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.return_layer is not None, return_dict=True)

        if self.return_layer is not None:
            z = outputs.hidden_states[self.return_layer]
            if self.do_final_ln:
                z = self.transformer.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state

        if self.inference_mode:
            original_mean = z.mean()
            batch_multipliers = batch_multipliers.to(z.device).to(z.dtype)
            z *= batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
            new_mean = z.mean()
            z *= original_mean / new_mean

        if future is not None:
            z = torch.cat((z, future), axis=-2)
            #print(z.shape)
        return z

    def encode(self, text):
        return self(text)

