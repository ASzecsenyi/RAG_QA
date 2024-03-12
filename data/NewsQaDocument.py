from typing import Any

from tqdm import tqdm

from data import Document
from datasets import load_dataset

import pandas as pd

newsqa = load_dataset('newsqa', data_dir='data/files')

newsqa_top_300 = [
    "./cnn/stories/aa88962e14a4b6218da615ed96254c695d46e182.story",
    "./cnn/stories/0e5b33b1b1c785437b431393ccfbc1bf0bf769e8.story",
    "./cnn/stories/4c90f974532ce6d3ff6f4f110394148df1147441.story",
    "./cnn/stories/815eec4729339dc3d8813e1a0a805d8a3c934532.story",
    "./cnn/stories/276a8e832e1948472da6524e0e513cca14b9afce.story",
    "./cnn/stories/c3b9b1e0151b0b1101634ea4c92b0045fe6d3c6d.story",
    "./cnn/stories/25645a2944b1ef1743d8b7582f6ede70aed84753.story",
    "./cnn/stories/9862b8aab2db9c82fd1012792783a90ec79f7269.story",
    "./cnn/stories/5bda86b0620b9e9e760b605bdc01d21a990650aa.story",
    "./cnn/stories/0c3ceb5d6b6ff38c90efd58d87c0b52da434232d.story",
    "./cnn/stories/a994137548230227d2d5bebe79782d9f7e728d2d.story",
    "./cnn/stories/1c1ff3af98f96893b4dbcf8c3b98aa195cc25ea8.story",
    "./cnn/stories/23f24292b366f81b1e4277d51e56f52644cc5b61.story",
    "./cnn/stories/a5e498f7832ee049566da21087ea58c12c41ce4c.story",
    "./cnn/stories/5e450811a0690a76c6bce0bc493e1050a3c5128a.story",
    "./cnn/stories/d3491d04a32a3587d52ee32ef3dda9852f57a14e.story",
    "./cnn/stories/e2730a7ce323f158d4ee672bb52f1214244fbad7.story",
    "./cnn/stories/d4dfbc7415e89e7e9df3710934c638cb2c7d4bb5.story",
    "./cnn/stories/ab50a080112b6bbcaa3166a05c1e369bb3289ccb.story",
    "./cnn/stories/5bf540c3df15a780eb231af6c1b6263eec08139f.story",
    "./cnn/stories/8403bab64569b5ea605aee4557f84a27465b3cf5.story",
    "./cnn/stories/ab532a5ce5df9786eddd67ce1fa4f4eeeb629979.story",
    "./cnn/stories/a8f6508736c136a450a79a10b42545e177a58c28.story",
    "./cnn/stories/493e1a795805fcf0de2aaed7e6cd66566d4e6972.story",
    "./cnn/stories/97ed47519371db40473f71f77a52a7beb5f0a7ef.story",
    "./cnn/stories/f873214b61f5870d03af8785c616fba82f28e810.story",
    "./cnn/stories/7c52867be2f99295337fe104d0464cf315217839.story",
    "./cnn/stories/8e4f05c96355f9b3089ee176ec5ddbbb28ce3f32.story",
    "./cnn/stories/c85a7eb698c39c7465d1fe4ebbb50447329b5497.story",
    "./cnn/stories/5806102cf9e813e48ced3314cea2a80e1bd614c7.story",
    "./cnn/stories/5b61e407e01856f861a9967908e592f8cabc0ed2.story",
    "./cnn/stories/35d5faf70a35e01e81d5cd4e0406dbe5cb1be10a.story",
    "./cnn/stories/6a0f9c8a5d0c6e8949b37924163c92923fe5770d.story",
    "./cnn/stories/2b2c8b817d4747dbb6f914a459fe3703736c420f.story",
    "./cnn/stories/94d6a496404c16cafed244b60fdda2c14b552244.story",
    "./cnn/stories/f2702feddad1dcaca1de453bf0b7c026f2b43eb5.story",
    "./cnn/stories/f3c1f061d06f13d358c64be34e16ec490e5a9135.story",
    "./cnn/stories/dac7a926543efef62b76f3e1b05f18c3edc01ac7.story",
    "./cnn/stories/9c296e3af6aa953b35fc903760d64f9b2d8c239f.story",
    "./cnn/stories/38e033b50aa3720540c9dae3cc5e4c622ff59312.story",
    "./cnn/stories/12ffb33bba8ed9fa1c5ab96150f04058b7739401.story",
    "./cnn/stories/4288f733a7149326dca1ad489f4bdefd3afc1f16.story",
    "./cnn/stories/a70424015a14f1adb0483bb006556e77411be371.story",
    "./cnn/stories/e29e772470903931c9cc53e0c19fd11e11efeb33.story",
    "./cnn/stories/2ff2a1326985d2a052e20cf25e5c3b3cf7a45d48.story",
    "./cnn/stories/d6402f873df3c15ae64859829679832cf767c64f.story",
    "./cnn/stories/933a9c1525a250e11e74311dedd9821596f831bf.story",
    "./cnn/stories/4264ba3d48a2fc7ea379a15a99f306917e5dc501.story",
    "./cnn/stories/b95239b85ed1c88d0d39035d1e7f948efe1e024d.story",
    "./cnn/stories/0ab642aea9c58f175c93dddec15071d1a0b2e792.story",
    "./cnn/stories/0b6d5cf34240aea1a6c8c6ff09ff5c91d0de6cfb.story",
    "./cnn/stories/1915dd69481b5f6033f91054ffdad42be380c215.story",
    "./cnn/stories/eab829ef92b6e3c70a2c0dbbffee718a063c979b.story",
    "./cnn/stories/6e0cf80b3ac1bb0d841ea81502f925b84fed0153.story",
    "./cnn/stories/aa9e0fb01f8e91c0034f26b978228ae8ee995554.story",
    "./cnn/stories/665725037e4f048d59c363c7b23f25e1ce9f6890.story",
    "./cnn/stories/5e7c990b12d43b077d476413a16c05fad2398c35.story",
    "./cnn/stories/bdf5569629d1627a5c56b06fbb5dabfba402e7e6.story",
    "./cnn/stories/dee351470f101ae014d080898b60ba56b09ba841.story",
    "./cnn/stories/b414366d7e9a533ec14b07b6b49518fe1e898bda.story",
    "./cnn/stories/a2008033ac3e22a5ab4203a06c59a014f46cacd2.story",
    "./cnn/stories/54c0e752966996569cfd4a89926f6f43717032dd.story",
    "./cnn/stories/61db1052501f599bd421ddb955bc40c15a84f48d.story",
    "./cnn/stories/d5abefb5bb35c6209db5993f33f1d40706a4fd10.story",
    "./cnn/stories/6ebacf0eea3263bc1c27fc7541c0725e1aebeafa.story",
    "./cnn/stories/e14c83e7db6e463da0a776d0c80c8c197de58772.story",
    "./cnn/stories/3087ea5447f9306d8c0e631101d9a9d2765435e1.story",
    "./cnn/stories/b8904dc43c7a7f4c2d99b53d6a6128424047efff.story",
    "./cnn/stories/a9ffab56869db5866de2d0ffe739ba624e55ce63.story",
    "./cnn/stories/23f6534b080425f53a09bb1973c7941c44a31dbf.story",
    "./cnn/stories/5c761544e3f9130dbc7f5d2018babeaf8cd7a3f5.story",
    "./cnn/stories/57fa9cae8fbd1e41cf54f150ebfed9a8ac925603.story",
    "./cnn/stories/a70e317f8f6ccd819213943de8c029775c8fc26b.story",
    "./cnn/stories/233071501603b10e60a5e44e38284c682bbb9240.story",
    "./cnn/stories/ef42653bf9337de184e74be0160b807a41b526fd.story",
    "./cnn/stories/517064c7afabd6c973f81455bf2c4eec5630a294.story",
    "./cnn/stories/55a6aabd120f0b18297f7efe22452f9b3aca0d8d.story",
    "./cnn/stories/5b5a3f82f235ebb073be860c7041fc3b7d66c84f.story",
    "./cnn/stories/9eed9be6ba6a5fe509c4ab53883916a6b4b085a0.story",
    "./cnn/stories/7e7d7f8f293e4750bc9a403cf8b46cda25fe616a.story",
    "./cnn/stories/98f6b33ff279a7ba3b0e214f121196a6c39ac184.story",
    "./cnn/stories/a07f9a04fb64984c0f9f6a4f7ee033386d9240b2.story",
    "./cnn/stories/62b94281e554fa5b4922383cbf71bc8d574eb129.story",
    "./cnn/stories/57b475ce1bc01cdd1e6621c681b4ea6089245fb3.story",
    "./cnn/stories/8919c342aede899547a5bf740bd801fefd4068d6.story",
    "./cnn/stories/ac63976c42516d2846b53d3e44728e2c1533d45e.story",
    "./cnn/stories/6c44741dad158b8349f0414ca2e5ec1fc9afb9bb.story",
    "./cnn/stories/23f84ace3ba30513e386f420db78a163c9483fac.story",
    "./cnn/stories/504f6f6e1aea6bb73c728d12a1342faa6828aace.story",
    "./cnn/stories/aabfda7d9f665e2fb375e46bd2b1d9b4092ed6b0.story",
    "./cnn/stories/f12e4bbb07211de7d43b4e331dc73404aa804562.story",
    "./cnn/stories/9ce5ebdae002aaa551e4bc5dd1e278991f20a2e6.story",
    "./cnn/stories/0534697013f80aeaebd88c82c051ad370d7c7c2e.story",
    "./cnn/stories/55461373100d4a36ff0097bdc676dfb02250c960.story",
    "./cnn/stories/9540de43c6177686202f1065f168c86b76ff8f0a.story",
    "./cnn/stories/ace7f6d9e984397bda16bf441019f1584f036b2a.story",
    "./cnn/stories/0d6937b789b603d702181fa67551c713d2fd6bd5.story",
    "./cnn/stories/e6153725110f97d0c88a8c2e09fb4126d6a9cc40.story",
    "./cnn/stories/7adb70b53090e070fdeab9429b789de268b1f1dd.story",
    "./cnn/stories/1ee14b83d22de97ed8354e2e943ec660ef5683df.story",
    "./cnn/stories/1f5ac3881d1b6710b607803fbcafd5030ab033cd.story",
    "./cnn/stories/db77c33da54ad40298d75740cc23f841e3c450c9.story",
    "./cnn/stories/182816fef64c4c2e58ece915551af64b04adbf53.story",
    "./cnn/stories/7db9975b2e4a1dc08a63b9a3b49e1df886eeceb1.story",
    "./cnn/stories/4579c95c002d50879235bfa10af344b56aa38964.story",
    "./cnn/stories/ca780d983e2d3abffb60d947401bc9c68ea49aab.story",
    "./cnn/stories/968f42754f880eae1626b8f327a9ae0b29468fb6.story",
    "./cnn/stories/ace8153406b2442cc5da8e60031921f2b16380ba.story",
    "./cnn/stories/11211aeb7c8f3d7a50c960bc6d148c1f168028be.story",
    "./cnn/stories/714782b8df2d0a8cde710805a44d819888771f88.story",
    "./cnn/stories/5b84008065eadb5e5307a5682cdc7ee64097a11e.story",
    "./cnn/stories/6ab14cfa16a0b6d7fbcda509193d394b4f2f56a0.story",
    "./cnn/stories/5a971b728a7e6447c618bd15f4263b7603d0a7a8.story",
    "./cnn/stories/0fca9cb001fd2f60ff7bacb76b3db287366b1cd1.story",
    "./cnn/stories/1e3f368df54a6551da3a51a628dacdbb307b33b6.story",
    "./cnn/stories/4595c85f7cb8c96f8b9647cfd7ecb6560ed39054.story",
    "./cnn/stories/2a6454e52077dfede0469e98deefee0f2ad8c264.story",
    "./cnn/stories/a8618ed361a940864ac34807e8912e25f3a00832.story",
    "./cnn/stories/c44cff38576542d432bdfa9b7ca9d2a27fd29b64.story",
    "./cnn/stories/5cc51e348579ac1cc83b89ddc46ec9f7424bb9ca.story",
    "./cnn/stories/92af98f512e0cbea87fd851954ff3c0b45693300.story",
    "./cnn/stories/d271bf998bac72de1e32a1ccd1c6221615558eb0.story",
    "./cnn/stories/e77d9a8ff1691c16f0dc0ce91c74776231bcc73f.story",
    "./cnn/stories/166e8a8af599b51323edf24f1aa27c46dbc93486.story",
    "./cnn/stories/e483ad70b03bb9184f105b5b1ce4a63b2b462320.story",
    "./cnn/stories/0d8a02341b9dec7fc9bdeb010321ca5c5ff324aa.story",
    "./cnn/stories/eac210df9ef39971d182d215e8a53228b0d4e578.story",
    "./cnn/stories/27d97c98917f7db0782a702be5992ef8cf6eef05.story",
    "./cnn/stories/2ffd3746d2f21464ff32a684957beeb8df7590f5.story",
    "./cnn/stories/5c19ebce58e729e01d8bf72b40b1670d86dbef76.story",
    "./cnn/stories/058ceacd672e573e736867c0a5bdd25061fbb400.story",
    "./cnn/stories/2d092badac54a4154b3029e78c67e07c41da77d0.story",
    "./cnn/stories/d4d6125d08422ca249497365a0a4ce4e70c75c6f.story",
    "./cnn/stories/ecc920c017bb7d3f671b5a3a3c0ed84f831c9c33.story",
    "./cnn/stories/9483ae6d83dd71a96e91eee727a9ce7c0f3cf6ab.story",
    "./cnn/stories/9c96ab22f0395ae016dc6ec1d8a9726d3ad434d2.story",
    "./cnn/stories/5cd8537d0d3fac20715f2fe4fa1c1fea05965839.story",
    "./cnn/stories/4db7b609fe32b226f1e6ed35d58ace52eb1dbd2a.story",
    "./cnn/stories/6f1e9eeab92c6e99658add76e44ef4a4cf3a1859.story",
    "./cnn/stories/5e5209fe7987f569c50cd69e684c9a982cff5108.story",
    "./cnn/stories/85a7aaa5b67a30872d4f47f0ec930465bfb442e8.story",
    "./cnn/stories/98ebd85013fb1cda2d8607bdaa15429c74e4f8ec.story",
    "./cnn/stories/7e3ff86334f50e758411041aa121f47c33319c8f.story",
    "./cnn/stories/e337f93e6d93f51fc69a5973e6ff614dd9ce6438.story",
    "./cnn/stories/b83deffd3acdc68afc9ab5a97cea5132eece50cd.story",
    "./cnn/stories/47aba666b72ded91a9a5d9889d9c97fa8d597e40.story",
    "./cnn/stories/91ff93e174c9aaa8db0c407b9f34d6fd9356d1be.story",
    "./cnn/stories/a28dac0f65d437701a18bb17cd997105d9ca9863.story",
    "./cnn/stories/70af269dacddc79f4a81c744b85b334a6f3124c3.story",
    "./cnn/stories/fab6925dffecf7712792d699f6f950fa4ceebb91.story",
    "./cnn/stories/e1090165e8c2a4d62ddbf75270705947824b08fa.story",
    "./cnn/stories/3efa3a9c43c46b668f86fdd79bb92481af5794a6.story",
    "./cnn/stories/d5a8cff95f73738ba66b31e5c718d096f0f327d8.story",
    "./cnn/stories/a7f578ca546ea016f5695fe3dd09f1b16e083575.story",
    "./cnn/stories/20c96fd45022c3255493eb1cf4884631aedd89f1.story",
    "./cnn/stories/c012c6f4390aac68ffc913e42fdb7e5c0cb3b770.story",
    "./cnn/stories/a6f1261be71711735e726353babe990f75ab284a.story",
    "./cnn/stories/8a0d6572d6d1b31400d0004e2ef3fa546be5fbe1.story",
    "./cnn/stories/bd96674919760ec2239ae4e4a7b42179228e97bd.story",
    "./cnn/stories/4f2f9c586a46a570347b3b5898dc193670737cf3.story",
    "./cnn/stories/e845dd89dd4837b1ae0fcc18b13e6b2f6285a270.story",
    "./cnn/stories/b8bea7b29fe95562984db42ba29f7053bf769f53.story",
    "./cnn/stories/2160c5d812611becf442b5079a7908c2f48f6de7.story",
    "./cnn/stories/1f516cdefd50e713cc67852e0c687b9deb5296db.story",
    "./cnn/stories/0e4f88645355e4cd0d574092b656af95c116d604.story",
    "./cnn/stories/26827f0c72d531ce979ef6ccea79a8b7b50d6db7.story",
    "./cnn/stories/a483adfdf3f345e702f6f8d67ee6e236f59e57ed.story",
    "./cnn/stories/9fa4a93d3c5f62e72c92cb662f7d44dbc8c26602.story",
    "./cnn/stories/538cb0c8900d3e5f2de349fb3a199441dba905e4.story",
    "./cnn/stories/8f6b7cc9ca78fb166179449f3ae6342a05ed7fbc.story",
    "./cnn/stories/420d8b930d57ee905c1f8cee167e06b1f96a5a40.story",
    "./cnn/stories/97c353ad5ce3cbd743aa2eaf9428714acceb52d0.story",
    "./cnn/stories/1c886637c5be85807b2397bb5a20dd017234e25f.story",
    "./cnn/stories/ff14c451cae58f4c6e73ecbfc64995ae4a5013a2.story",
    "./cnn/stories/db43ca754ff385ab9ead775b981d248924c7a4a1.story",
    "./cnn/stories/643d86992d1ce964ed055b2ef4aee58a2cf88223.story",
    "./cnn/stories/45901954a9c7ab101eab7490c9d25063f9a11a77.story",
    "./cnn/stories/ca2803d52e53efd9be9f5be27ef0497a314287ef.story",
    "./cnn/stories/8b6e9daa35cd41136539e4f8a620e62a87e9cf30.story",
    "./cnn/stories/bd63808625b2a7f02ca00e44b6f24e1e3b162647.story",
    "./cnn/stories/fa10b80d47d6d0877a97ce1f00f4e81f245a7319.story",
    "./cnn/stories/7f41da57205facdf1d95243ced8eb7eca81ff31b.story",
    "./cnn/stories/639c56d5333032b8295dc33f2292de4fb0f63619.story",
    "./cnn/stories/63ba648eb30a301f1cf9045f80a3ab2aa21eb07b.story",
    "./cnn/stories/16891819e12c08ca1c7c1e279b401f57d67c5911.story",
    "./cnn/stories/9b986a44799c41672d4d4c364a3076359eaaa820.story",
    "./cnn/stories/8ce4fb3457776fd2fa88a9b529c65343d37dc05c.story",
    "./cnn/stories/fb017e38cc560e418786c9f4e443f5668bf9eb75.story",
    "./cnn/stories/a3f316275917b6826683f7e9e620060ee5b8f9c7.story",
    "./cnn/stories/5b5825799efedb1dc8757ba4a0d99de7915dc856.story",
    "./cnn/stories/c9414bf55859cb9ca7e925a4a09b416b6f66446a.story",
    "./cnn/stories/7c72b57e65ddefc77362ed64f7dbc0aebb87ac81.story",
    "./cnn/stories/37d1bab55ca378c5fe777a7697a69f7e8d2e5101.story",
    "./cnn/stories/9c4199f76e1f65d425eb53b37a68ef92e93e5fb2.story",
    "./cnn/stories/b5e1cdebe26c799c67286a5d3b3d0713100b7a1c.story",
    "./cnn/stories/d68720b4dcdf00ebdf6e099409b03aec1e332bf0.story",
    "./cnn/stories/0ffc96773f598d415036b1a17797863483a091c6.story",
    "./cnn/stories/3eb68f244f9159e05cfe40316ac9e27fc938cf3a.story",
    "./cnn/stories/ce60e601346cec5610389a7d3d6604175df003a0.story",
    "./cnn/stories/eeef836fec82befa725a09b8028c0de529dd6ad0.story",
    "./cnn/stories/7302a541259f8667d8ce4342a313404f442343dd.story",
    "./cnn/stories/ffbd623d043e1aedd1d2602d642f22ae497fb907.story",
    "./cnn/stories/ca012876b5b6faf014d03f46e0e0537cc4fddfd0.story",
    "./cnn/stories/cd6d005cf83ba0db54cb13480406bd94b3ad5a1f.story",
    "./cnn/stories/438995888712f35ae4d428e9a78d8abb564cd6a7.story",
    "./cnn/stories/1905a821fef6a6cc8adc63bd64628eab3c82306b.story",
    "./cnn/stories/edfd5023aaca8b6570286cfb31b2e61d9d3d8476.story",
    "./cnn/stories/9bd5a4a840c45c51ba1117bef4e75a83054b1cff.story",
    "./cnn/stories/d80e6a3be826df05df00a87f49cc426fd597f085.story",
    "./cnn/stories/76f04a2f8b68a6aca560a7118eacff4020088537.story",
    "./cnn/stories/f42d05545d8b1840f495c14fd7a43fcb48ab389f.story",
    "./cnn/stories/19e0bec2f73b2be9e423a8f1639aa7a374554474.story",
    "./cnn/stories/54ffe8a16c505610dbce10c456421234df20c4b2.story",
    "./cnn/stories/0c30f8b99da29f37d4151b54e279d9eec2fb2a8d.story",
    "./cnn/stories/c8685e38369975669446656ee5819a6f0c39b5a2.story",
    "./cnn/stories/74999197b3f961d79bc0602e675f4032a053002d.story",
    "./cnn/stories/4d4ea58f0771c94c9c3ba8c2f844a15ef44e5937.story",
    "./cnn/stories/6351e66d90505c85abd4e7b3eda8677aeaeeede4.story",
    "./cnn/stories/b27c6bccb305b4f410f1531bc9f5cd174651c7ba.story",
    "./cnn/stories/74050867f3fbe3043aba1f0ead4636a9ffd788ee.story",
    "./cnn/stories/b494fc1ae7aa77392dfe5e7f395aeaa30d50c87e.story",
    "./cnn/stories/6390895126471a970b9eeb099f1c1a0c2930a2fa.story",
    "./cnn/stories/7dae6ed91c0b5a94845e798b97728b08d82462d1.story",
    "./cnn/stories/551f3d88d05567e87b91bd9b222335605867604f.story",
    "./cnn/stories/eed527dece78deffed54a7cdd8516c4d57a90011.story",
    "./cnn/stories/9b929e9396a07763964314c994289a18590a5239.story",
    "./cnn/stories/677112677907410588ed78384ec10353a21de83d.story",
    "./cnn/stories/71d91b5a77a53fcc7ffb7fb613ebd84c25b308a8.story",
    "./cnn/stories/bc20d20b013a0415da2f96a9b49e91639c55ebc7.story",
    "./cnn/stories/76894b1e3bff1217bb5a11ee880a378426942a49.story",
    "./cnn/stories/831005755f85012c882f17c3b3699b34a7febb7b.story",
    "./cnn/stories/937717a38bf1a174febfd009a9bc991d54d3ce6e.story",
    "./cnn/stories/c1627a365e316b432078893d1c2a779c84848dba.story",
    "./cnn/stories/b6ebbe378c675ab15eb73712200f484b2e3e7cd4.story",
    "./cnn/stories/5bfcd0557beeb60277ef705f4f6ad1ceac11ffa0.story",
    "./cnn/stories/7aa4c007af6f36015226ed01ee5a54ef549e6ed5.story",
    "./cnn/stories/152764841ded25def44c8726018136d5b62d541e.story",
    "./cnn/stories/117050c9ae0877c9e1b5a3d0d874ffcc5d271710.story",
    "./cnn/stories/1915c9eb1b7b1ba748827b821561b1eaa97948b8.story",
    "./cnn/stories/896feee1842982cacfa0a022b5901a6a2d64ef68.story",
    "./cnn/stories/279e955a0e3800800d439a9ce9a683a017730d2c.story",
    "./cnn/stories/43938c17b80c51879f77bfa155263214dfdff223.story",
    "./cnn/stories/28b628786a5a82f46a2ed77db4bafa3f4b956719.story",
    "./cnn/stories/b5e30289750e45cb118b671eef6314cf3f3245ee.story",
    "./cnn/stories/dfb1a8f4be3ef9a338c7d75c75e7dbfbb18dad86.story",
    "./cnn/stories/4a9a0e027f1620208daa3c558658f344cad4ba20.story",
    "./cnn/stories/5957de81c7c1d82f9479617f08daec8bc8b48c39.story",
    "./cnn/stories/b4782086e0c76634f7f67507d8918e7289162c0f.story",
    "./cnn/stories/eefcfa7320d31de203071a72428efd7e64324611.story",
    "./cnn/stories/8180a77657661e975e3fac6c7fdbd8de97fd160e.story",
    "./cnn/stories/8e596db6b60963ef715da3fa289e4ad495d047bb.story",
    "./cnn/stories/ab3b983aec43643056b735ba1d2d354342fc63f5.story",
    "./cnn/stories/3c20137f3f7403ce3ef18ec543f12cfe030b711b.story",
    "./cnn/stories/9ebbf764d627ed60c66c05297178900755508951.story",
    "./cnn/stories/a0dbd0e34f9bdec7924f96e66e32f1d6970e3876.story",
    "./cnn/stories/dacfa443475efb6c9bfc96c2c44eb52691db24de.story",
    "./cnn/stories/6a82b88de0ba92b80e6aa536ce748b0bb03fc7ba.story",
    "./cnn/stories/c0a9f5ca1bdc47073d953449715029121b654c27.story",
    "./cnn/stories/cf08d87683e8c03289b7dfd87c60d98246270be2.story",
    "./cnn/stories/cd59be1b2a756ac2ac87156560e534e90a04ad32.story",
    "./cnn/stories/62807965f127049477a80f09fecd3188d5d01413.story",
    "./cnn/stories/5bb09b493dc5f30b1b2c999c9dcb580ee13f662a.story",
    "./cnn/stories/1d7e3225e1913b841440a6837fc02ca3f5e9aec6.story",
    "./cnn/stories/f6f46aa872450faa3ba4679c75840cb5bd05a2cc.story",
    "./cnn/stories/58a9f6b64d34bb12016a672c583f0f34082e8597.story",
    "./cnn/stories/f7709df3257aafd7b4f8f4e7bce939b98b63951c.story",
    "./cnn/stories/0254b59405a4bd611d20137ba4ffa579a3f872d1.story",
    "./cnn/stories/e4407bbedabb72aaae7a8b63f01309de6a27ba54.story",
    "./cnn/stories/44c0a4109d944f279742cd5a89773af3eebc7c65.story",
    "./cnn/stories/fc8ff49f5e15165328f08b4bb930391db7628130.story",
    "./cnn/stories/9a6b998ae3696033bbdf4973a540c8e843c7001b.story",
    "./cnn/stories/160228fe18f272761b969b08622ba868ad65b206.story",
    "./cnn/stories/a98c702bea39a40fea315d1c34c90d311bf01b63.story",
    "./cnn/stories/e19aa8bcd13b10380ae048655d101be0d1705424.story",
    "./cnn/stories/3d6f02e5a73f1248c9274af8358f3ad010c820f0.story",
    "./cnn/stories/e8a9cc12b234ee8f2fd05f74b344ce0c166f09d0.story",
    "./cnn/stories/e6e5b19179a86490b071beb2a239f07086948b99.story",
    "./cnn/stories/afc3503c2ba8eeafb6b1a39e47439434e5413348.story",
    "./cnn/stories/6ccc90832425c1590001a0fa8ef77e42f5516b02.story",
    "./cnn/stories/009e2c2f7a557cd6c431255707cf863f617076d6.story",
    "./cnn/stories/e2d72830b04ed5d5774158e52f66cfe8ed723ad3.story",
    "./cnn/stories/7551503e7e57bd519913b0df90ca1e80d5305b05.story",
    "./cnn/stories/964f9d750403e6c7d43a24da887c1338ab36e9e9.story",
    "./cnn/stories/411a38fb5216888d83f1b6004aef76a70139768a.story",
    "./cnn/stories/f5a1a323b8f3beb6a64b4ab4f42a254209c037cd.story",
    "./cnn/stories/8154b09ae15bef8f3a9092e1b020f1a86a57dc28.story",
    "./cnn/stories/95f10ce802d3670dde1d50d3cda7a2eaf614bced.story",
    "./cnn/stories/d410d413e7a71ca6e1046641eb9c8c3b816c174f.story",
    "./cnn/stories/6f0d03298efefcb24aaed05ac51e084cff13d8f1.story",
    "./cnn/stories/ddd2c82e73bb6e11ba93babccc3585476114305c.story",
    "./cnn/stories/e4794151cef68cf90472edc9229711e7572cfa11.story",
    "./cnn/stories/4990652aa626e4cddf8497aa08ef211003b6b1b5.story",
    "./cnn/stories/d71bbc5ad05f2f8a39c1cfe0c0e82a48aa2779ce.story",
    "./cnn/stories/302e5c8d47851263ca27291a760836b25de2c0dd.story",
    "./cnn/stories/b53754c54be75724270ef37ef9643d79b53185b8.story",
    "./cnn/stories/272e13ef7a64d86ddd23d31042a89e95193a9989.story",
    "./cnn/stories/965921073e4607b47883e08d522ad5377c6b332e.story",
    "./cnn/stories/e6df303f8414574b7e297bf65e96ee3479ef3fc7.story",
    "./cnn/stories/ba670306341e8bcc53da274572456affdbb2aca7.story",
    "./cnn/stories/1ea7971780538f433c42704ccb6e94e2cb1546c2.story",
]


class NewsQaDocument(Document):
    """
    A document from the NewsQA dataset.
    """

    def __init__(self, name: str = None, story_id: str = None, split: str = 'test'):
        """
        :param name: the document identifier
        :type name: str, optional
        :param story_id: the id of the story to get data from
        :type story_id: str
        """

        if story_id is None:
            story_id = newsqa[split][0]['story_id']

        if name is None:
            name = f"newsqa_{story_id}"

        print(name)

        # get all instances of the story and questions about it
        story_questions = newsqa[split].filter(lambda x: x['story_id'] == story_id)

        assert len(story_questions) > 0, "story_id must be in the dataset"

        # it is assumed that all story_text instances with the same story_id are the same

        document = story_questions['story_text'][0]

        questions: list[dict[str, Any]] = []
        for question in story_questions:
            # get texts at the answer token ranges
            answer_token_ranges = [
                [int(token_range.split(':')[0]), int(token_range.split(':')[1])]
                for token_range in question['answer_token_ranges'].split(',')]
            document_tokens = document.split(' ')
            ground_truths = [' '.join(document_tokens[start:end]) for start, end in answer_token_ranges]

            questions.append(
                {
                    "question": question['question'],
                    "answer_token_ranges": question['answer_token_ranges'],
                    "ground_truths": ground_truths,
                }
            )

        super().__init__(document, name, questions)

    @classmethod
    def all_documents(cls, split: str = 'test', max_stories: int = None) -> list["NewsQaDocument"]:
        """
        Creates a list of all unique documents in the dataset.

        :return: the list of documents
        :rtype: list[Document]
        :param split: the split to get documents from
        :type split: str
        :param max_stories: the maximum number of documents to get, defaults to None - gets all documents
        :type max_stories: int, optional
        """

        documents = []

        unique_story_ids = pd.unique(newsqa[split]['story_id'])

        if max_stories is not None:
            unique_story_ids = unique_story_ids[:max_stories]

        for story_id in tqdm(unique_story_ids):
            try:
                documents.append(cls(story_id=story_id, split=split))
            except ValueError as e:
                print(e)
        return documents

