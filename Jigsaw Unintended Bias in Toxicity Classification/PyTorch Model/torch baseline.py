from contextlib import contextmanager
import os, random, re, string, time, warnings, gc, math
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from nltk.tokenize.treebank import TreebankWordTokenizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.utils.data
from torch.optim.optimizer import Optimizer

# EMBEDDING_FASTTEXT = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
# EMBEDDING_GLOVE = '../input/glove840b300dtxt/glove.840B.300d.txt'
CRAWL_EMBEDDING_PATH = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'
GLOVE_EMBEDDING_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'
TRAIN_DATA = '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
TEST_DATA = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
SAMPLE_SUBMISSION = '../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv'
# EMBEDDING_FILES = [EMBEDDING_FASTTEXT, EMBEDDING_GLOVE]
EMBEDDING_FILES = [CRAWL_EMBEDDING_PATH, GLOVE_EMBEDDING_PATH]

embed_size = 300
max_features = 400000
maxlen = 220

batch_size = 1024
n_epochs = 5
n_splits = 5
seed = 1024
lr = 0.001

@contextmanager
def timer(msg):
    t0 = time.time()
    print(f'[{msg}] start.')
    yield
    elapsed_time = time.time() - t0
    print(f'[{msg}] done in {elapsed_time / 60:.2f} min.')

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

symbols_to_isolate = '.,?!-;*"…:—()%#$&_/@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁'
symbols_to_delete = '\n🍕\r🐵😑\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'

treetokenizer = TreebankWordTokenizer()
isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}
remove_dict = {ord(c):f'' for c in symbols_to_delete}

def handle_punctuation(x):
    x = x.translate(remove_dict)
    x = x.translate(isolate_dict)
    return x

def handle_contractions(x):
    x = treetokenizer.tokenize(x)
    return x

def fix_quote(x):
    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
    x = ' '.join(x)
    return x

def preprocess(x):
    x = handle_punctuation(x)
    x = handle_contractions(x)
    x = fix_quote(x)
    return x
        
# def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

# def load_embeddings(embed_dir = None):
#     embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embed_dir))
#     return embedding_index

# def build_matrix(word_index, embedding_file):
#     embeddings_index = load_embeddings(embed_dir = embedding_file)
#     embedding_matrix = np.zeros((len(word_index) + 1,300))
#     for word, i in word_index.items():
#         try:
#             embedding_matrix[i] = embeddings_index[word]
#         except:
#             embedding_matrix[i] = embeddings_index["unknown"]
    
#     del embeddings_index; gc.collect()
#     return embedding_matrix
    
def load_embeddings(path):
    with open(path,'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr
    
def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((max_features + 1, 300))
    unknown_words = []
    
    for word, i in word_index.items():
        if i <= max_features:
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                try:
                    embedding_matrix[i] = embedding_index[word.lower()]
                except KeyError:
                    try:
                        embedding_matrix[i] = embedding_index[word.title()]
                    except KeyError:
                        unknown_words.append(word)
                        
    return embedding_matrix
    
def custom_loss(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])
    
    return bce_loss_1 + bce_loss_2
    
def add_num_feats(df):
    df["comment_length"] = df["comment_text"].apply(len)
    df["capitals"] = df["comment_text"].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df["cap_vs_length"] = df.apply(lambda row: float(row["capitals"])/float(row["comment_length"]), axis = 1)
    df["num_words"] = df.comment_text.str.count("\S+")
    df["num_unique_words"] = df["comment_text"].apply(lambda comment: len(set(w for w in comment.split())))
    df["words_vs_unique"] = df["num_unique_words"]/df["num_words"]
    
    return df
    
# competition metric
def power_mean(x, p=-5):
    return np.power(np.mean(np.power(x, p)),1/p)
    
def get_s_auc(y_true,y_pred,y_identity):
    mask = y_identity==1
    s_auc = roc_auc_score(y_true[mask],y_pred[mask])
    return s_auc

def get_bpsn_auc(y_true,y_pred,y_identity):
    mask = (y_identity==1) & (y_true==0) | (y_identity==0) & (y_true==1)
    bpsn_auc = roc_auc_score(y_true[mask],y_pred[mask])
    return bpsn_auc

def get_bspn_auc(y_true,y_pred,y_identity):
    mask = (y_identity==1) & (y_true==1) | (y_identity==0) & (y_true==0)
    bspn_auc = roc_auc_score(y_true[mask],y_pred[mask])
    return bspn_auc

def get_total_auc(y_true,y_pred,y_identities):
    N = y_identities.shape[1]
    saucs = np.array([get_s_auc(y_true,y_pred,y_identities[:,i]) for i in range(N)])
    bpsns = np.array([get_bpsn_auc(y_true,y_pred,y_identities[:,i]) for i in range(N)])
    bspns = np.array([get_bspn_auc(y_true,y_pred,y_identities[:,i]) for i in range(N)])

    M_s_auc = power_mean(saucs)
    M_bpsns_auc = power_mean(bpsns)
    M_bspns_auc = power_mean(bspns)
    rauc = roc_auc_score(y_true,y_pred)

    total_auc = M_s_auc + M_bpsns_auc + M_bspns_auc + rauc
    total_auc/= 4

    return total_auc
    
class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        
    def register(self, name, val):
        self.shadow[name] = val.clone()
        
    def __call__(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

# code inspired from: https://github.com/anandsaha/pytorch.cyclic.learning.rate/blob/master/cls.py
class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs
        
class AdamW(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay using the method from
            the paper `Fixing Weight Decay Regularization in Adam` (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Fixing Weight Decay Regularization in Adam:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        self._initial_lr = lr

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    #eta = group['lr'] / self._initial_lr # scheduler changes lr only
                    #p.data.add_(-group['weight_decay'] * eta, p.data)
                    w = group['weight_decay'] * group['lr']
                    p.data.add_(-w, p.data)

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
        
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
        
class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix):
        super(NeuralNet, self).__init__()

        lstm_hidden_size = 64
        gru_hidden_size = 64
        dense_hidden_size = 6*gru_hidden_size
        self.gru_hidden_size = gru_hidden_size

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.25)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], lstm_hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(lstm_hidden_size * 2, gru_hidden_size, bidirectional=True, batch_first=True)
        
        self.linear = nn.Linear(dense_hidden_size, dense_hidden_size)
        self.relu = nn.ReLU()
        self.hidden_bn = nn.BatchNorm1d(dense_hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.target_out = nn.Linear(dense_hidden_size, 1)
        self.aux_out = nn.Linear(dense_hidden_size, 6)
        
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, hh_gru = self.gru(h_lstm)

        hh_gru = hh_gru.view(-1, self.gru_hidden_size * 2)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)
        
        conc = torch.cat((hh_gru, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.hidden_bn(conc)
        target_out = self.target_out(conc)
        aux_out = self.aux_out(conc)
        out = torch.cat([target_out, aux_out], 1)
        
        return out

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def load_and_prec():
    train = pd.read_csv(TRAIN_DATA, index_col='id')
    test = pd.read_csv(TEST_DATA, index_col='id')
    
    # down sampling
    # train["article_count"] = train["article_id"].map(train["article_id"].value_counts())
    # train = train[train["article_count"] <= 600]
    
    # clean the text
    train['comment_text'] = train['comment_text'].apply(lambda x: preprocess(x))
    test['comment_text'] = test['comment_text'].apply(lambda x: preprocess(x))
    
    # fill up the missing values
    train_x = train['comment_text'].fillna('_##_').values
    test_x = test['comment_text'].fillna('_##_').values
    
    identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish','muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    y_identities = (train[identity_columns] >= 0.5).astype(int).values
    
    # Overall
    weights = np.ones((len(train_x),)) / 4
    # Subgroup
    weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
    # Background Positive, Subgroup Negative
    weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +
    (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
    # Background Negative, Subgroup Positive
    weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +
       (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
    
    # tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features, filters = "", lower = False)
    tokenizer.fit_on_texts(list(train_x))
    train_x = tokenizer.texts_to_sequences(train_x)
    test_x = tokenizer.texts_to_sequences(test_x)
    
    # pad the sentences
    train_x = pad_sequences(train_x, maxlen=maxlen)
    test_x = pad_sequences(test_x, maxlen=maxlen)
    
    # get the target values
    train_y = np.vstack([train['target'].values, weights]).T
    train_aux_y = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values 

    # shuffling the data
    np.random.seed(seed)
    train_idx = np.random.permutation(len(train_x))

    train_x = train_x[train_idx]
    train_y = train_y[train_idx]
    train_aux_y = train_aux_y[train_idx]
    y_identities = y_identities[train_idx]
    
    return train_x, np.hstack([train_y, train_aux_y]), test_x, tokenizer.word_index, y_identities

def main():
    warnings.filterwarnings('ignore')
    
    with timer("load data"):
        train_x, train_y, test_x, word_index, y_identities = load_and_prec()
        
    with timer("load embedding matrix"):
        embedding_matrix = np.concatenate([build_matrix(word_index, embedding_file)
                                            for embedding_file in EMBEDDING_FILES], axis = -1)
        print(embedding_matrix.shape)
    with timer("train"):
        train_preds = np.zeros((len(train_x), 1))
        test_preds = np.zeros((len(test_x), 1))
    
        seed_torch(seed)
        
        x_test_cuda = torch.tensor(test_x, dtype=torch.long).cuda()
        test = torch.utils.data.TensorDataset(x_test_cuda)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    
        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(train_x, np.where(train_y[:, 0] >= 0.5, 1, 0)))
        
        for fold, (train_idx, valid_idx) in enumerate(splits):
            print("-"*80)
            print(f'Training fold {fold + 1}...')
            
            check_point = "pytorch_model_{}_fold".format(fold)
            valid_fold_preds = []
            test_fold_preds = []
            checkpoint_weights = [2**epoch for epoch in range(n_epochs-2)]
            
            x_train_fold = torch.tensor(train_x[train_idx], dtype=torch.long).cuda()
            y_train_fold = torch.tensor(train_y[train_idx], dtype=torch.float).cuda()

            x_val_fold = torch.tensor(train_x[valid_idx], dtype=torch.long).cuda()
            y_val_fold = torch.tensor(train_y[valid_idx], dtype=torch.float).cuda()
            
            train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
            valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
    
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
            
            model = NeuralNet(embedding_matrix).cuda()

            # loss_fn = torch.nn.BCEWithLogitsLoss(reduction = "mean")
            param_lrs = [{"params": param, "lr": lr} for param in model.parameters()]
            # optimizer = torch.optim.Adam(param_lrs, lr=lr)
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.5**epoch)
            scheduler = CyclicLR(optimizer, base_lr=lr, max_lr=0.005,
                                step_size=4*int(len(train_idx)/batch_size), mode='exp_range',
                                gamma=0.99994)
                
            # Exponential Moving Average
            ema = EMA(0.35)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema.register(name, param.data)
                    
            for epoch in range(n_epochs):
                start_time = time.time()
                # scheduler.step()
                model.train()
                avg_loss = 0.
    
                for i, (x_batch, y_batch) in enumerate(train_loader):
                    y_pred = model(x_batch)
                    # loss = loss_fn(y_pred, y_batch)
                    scheduler.batch_step()
                    loss = custom_loss(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            param.data = ema(name, param.data)
                    avg_loss += loss.item() / len(train_loader)
                
                model.eval()
                valid_preds_fold = np.zeros((len(valid_idx), 1))
                test_preds_fold = np.zeros((len(test_x), 1))
                avg_val_loss = 0.
    
                for i, (x_batch, y_batch) in enumerate(valid_loader):
                    with torch.no_grad():
                        y_pred = model(x_batch).detach()
                    avg_val_loss += custom_loss(y_pred, y_batch).item() / len(valid_loader)
                    valid_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, :1]
    
                elapsed_time = time.time() - start_time
                print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
    
                for i, (x_batch,) in enumerate(test_loader):
                    with torch.no_grad():
                        y_pred = model(x_batch).detach()
                    test_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, :1]
                
                if epoch + 1 > 2: 
                    valid_fold_preds.append(valid_preds_fold)
                    test_fold_preds.append(test_preds_fold)
                    
            torch.save(model.state_dict(), check_point)    
            train_preds[valid_idx] = np.average(valid_fold_preds, weights = checkpoint_weights, axis = 0)
            test_preds += np.average(test_fold_preds, weights =  checkpoint_weights, axis = 0) / n_splits
            print(f'fold {fold+1} cv score: {get_total_auc(np.where(train_y[valid_idx, 0] >= 0.5, 1, 0), valid_preds_fold, y_identities[valid_idx]):<8.5f}')
            print(f'fold {fold+1} cv score: {roc_auc_score(np.where(train_y[valid_idx, 0] >= 0.5, 1, 0), valid_preds_fold)}')
        
        print("-"*80)
        print(f'cv score: {get_total_auc(np.where(train_y[:, 0] >= 0.5, 1, 0), train_preds, y_identities):<8.5f}')
        print(f'roc auc score: {roc_auc_score(np.where(train_y[:, 0] >= 0.5, 1, 0), train_preds):<8.5f}')
        
    with timer('submit'):
        submission = pd.read_csv(SAMPLE_SUBMISSION, index_col='id')
        submission['prediction'] = test_preds
        submission.reset_index(drop=False, inplace=True)
        submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
    
