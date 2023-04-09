pyinstaller --python=/usr/local/bin/python3.8 \
--add-data "./streamlink/plugins/*:streamlink/plugins" \
--add-data "./streamlink/*:streamlink" \
--hidden-import argparse \
--hidden-import=sip --paths=/usr/local/lib/python3.8/site-packages/sipbuild \
--hidden-import streamlink.plugins.abematv \
--hidden-import streamlink.plugins.adultswim \
--hidden-import streamlink.plugins.afreeca \
--hidden-import streamlink.plugins.albavision \
--hidden-import streamlink.plugins.aloula \
--hidden-import streamlink.plugins.app17 \
--hidden-import streamlink.plugins.ard_live \
--hidden-import streamlink.plugins.ard_mediathek \
--hidden-import streamlink.plugins.artetv \
--hidden-import streamlink.plugins.atpchallenger \
--hidden-import streamlink.plugins.atresplayer \
--hidden-import streamlink.plugins.bbciplayer \
--hidden-import streamlink.plugins.bfmtv \
--hidden-import streamlink.plugins.bigo \
--hidden-import streamlink.plugins.bilibili \
--hidden-import streamlink.plugins.blazetv \
--hidden-import streamlink.plugins.bloomberg \
--hidden-import streamlink.plugins.booyah \
--hidden-import streamlink.plugins.brightcove \
--hidden-import streamlink.plugins.btv \
--hidden-import streamlink.plugins.cbsnews \
--hidden-import streamlink.plugins.cdnbg \
--hidden-import streamlink.plugins.ceskatelevize \
--hidden-import streamlink.plugins.cinergroup \
--hidden-import streamlink.plugins.clubbingtv \
--hidden-import streamlink.plugins.cmmedia \
--hidden-import streamlink.plugins.cnews \
--hidden-import streamlink.plugins.crunchyroll \
--hidden-import streamlink.plugins.dailymotion \
--hidden-import streamlink.plugins.dash \
--hidden-import streamlink.plugins.delfi \
--hidden-import streamlink.plugins.deutschewelle \
--hidden-import streamlink.plugins.dlive \
--hidden-import streamlink.plugins.dogan \
--hidden-import streamlink.plugins.dogus \
--hidden-import streamlink.plugins.drdk \
--hidden-import streamlink.plugins.earthcam \
--hidden-import streamlink.plugins.euronews \
--hidden-import streamlink.plugins.facebook \
--hidden-import streamlink.plugins.filmon \
--hidden-import streamlink.plugins.foxtr \
--hidden-import streamlink.plugins.galatasaraytv \
--hidden-import streamlink.plugins.goltelevision \
--hidden-import streamlink.plugins.goodgame \
--hidden-import streamlink.plugins.googledrive \
--hidden-import streamlink.plugins.gulli \
--hidden-import streamlink.plugins.hiplayer \
--hidden-import streamlink.plugins.hls \
--hidden-import streamlink.plugins.http \
--hidden-import streamlink.plugins.htv \
--hidden-import streamlink.plugins.huajiao \
--hidden-import streamlink.plugins.huya \
--hidden-import streamlink.plugins.idf1 \
--hidden-import streamlink.plugins.invintus \
--hidden-import streamlink.plugins.kugou \
--hidden-import streamlink.plugins.linelive \
--hidden-import streamlink.plugins.livestream \
--hidden-import streamlink.plugins.lnk \
--hidden-import streamlink.plugins.lrt \
--hidden-import streamlink.plugins.ltv_lsm_lv \
--hidden-import streamlink.plugins.mdstrm \
--hidden-import streamlink.plugins.mediaklikk \
--hidden-import streamlink.plugins.mediavitrina \
--hidden-import streamlink.plugins.mildom \
--hidden-import streamlink.plugins.mitele \
--hidden-import streamlink.plugins.mixcloud \
--hidden-import streamlink.plugins.mjunoon \
--hidden-import streamlink.plugins.mrtmk \
--hidden-import streamlink.plugins.n13tv \
--hidden-import streamlink.plugins.nbcnews \
--hidden-import streamlink.plugins.nhkworld \
--hidden-import streamlink.plugins.nicolive \
--hidden-import streamlink.plugins.nimotv \
--hidden-import streamlink.plugins.nos \
--hidden-import streamlink.plugins.nownews \
--hidden-import streamlink.plugins.nrk \
--hidden-import streamlink.plugins.ntv \
--hidden-import streamlink.plugins.okru \
--hidden-import streamlink.plugins.olympicchannel \
--hidden-import streamlink.plugins.oneplusone \
--hidden-import streamlink.plugins.onetv \
--hidden-import streamlink.plugins.openrectv \
--hidden-import streamlink.plugins.pandalive \
--hidden-import streamlink.plugins.picarto \
--hidden-import streamlink.plugins.piczel \
--hidden-import streamlink.plugins.pixiv \
--hidden-import streamlink.plugins.pluto \
--hidden-import streamlink.plugins.pluzz \
--hidden-import streamlink.plugins.qq \
--hidden-import streamlink.plugins.radiko \
--hidden-import streamlink.plugins.radionet \
--hidden-import streamlink.plugins.raiplay \
--hidden-import streamlink.plugins.reuters \
--hidden-import streamlink.plugins.rtbf \
--hidden-import streamlink.plugins.rtpa \
--hidden-import streamlink.plugins.rtpplay \
--hidden-import streamlink.plugins.rtve \
--hidden-import streamlink.plugins.rtvs \
--hidden-import streamlink.plugins.ruv \
--hidden-import streamlink.plugins.sbscokr \
--hidden-import streamlink.plugins.showroom \
--hidden-import streamlink.plugins.sportal \
--hidden-import streamlink.plugins.sportschau \
--hidden-import streamlink.plugins.ssh101 \
--hidden-import streamlink.plugins.stadium \
--hidden-import streamlink.plugins.steam \
--hidden-import streamlink.plugins.streamable \
--hidden-import streamlink.plugins.streann \
--hidden-import streamlink.plugins.stv \
--hidden-import streamlink.plugins.svtplay \
--hidden-import streamlink.plugins.swisstxt \
--hidden-import streamlink.plugins.telefe \
--hidden-import streamlink.plugins.tf1 \
--hidden-import streamlink.plugins.trovo \
--hidden-import streamlink.plugins.turkuvaz \
--hidden-import streamlink.plugins.tv360 \
--hidden-import streamlink.plugins.tv3cat \
--hidden-import streamlink.plugins.tv4play \
--hidden-import streamlink.plugins.tv5monde \
--hidden-import streamlink.plugins.tv8 \
--hidden-import streamlink.plugins.tv999 \
--hidden-import streamlink.plugins.tvibo \
--hidden-import streamlink.plugins.tviplayer \
--hidden-import streamlink.plugins.tvp \
--hidden-import streamlink.plugins.tvrby \
--hidden-import streamlink.plugins.tvrplus \
--hidden-import streamlink.plugins.tvtoya \
--hidden-import streamlink.plugins.twitcasting \
--hidden-import streamlink.plugins.twitch \
--hidden-import streamlink.plugins.useetv \
--hidden-import streamlink.plugins.ustreamtv \
--hidden-import streamlink.plugins.ustvnow \
--hidden-import streamlink.plugins.vidio \
--hidden-import streamlink.plugins.vimeo \
--hidden-import streamlink.plugins.vinhlongtv \
--hidden-import streamlink.plugins.vkplay \
--hidden-import streamlink.plugins.vk \
--hidden-import streamlink.plugins.vlive \
--hidden-import streamlink.plugins.vtvgo \
--hidden-import streamlink.plugins.wasd \
--hidden-import streamlink.plugins.webtv \
--hidden-import streamlink.plugins.welt \
--hidden-import streamlink.plugins.wwenetwork \
--hidden-import streamlink.plugins.youtube \
--hidden-import streamlink.plugins.yupptv \
--hidden-import streamlink.plugins.zattoo \
--hidden-import streamlink.plugins.zdf_mediathek \
--hidden-import streamlink.plugins.zeenews \
--hidden-import streamlink.plugins.zengatv \
--hidden-import streamlink.plugins.zhanqi \
--additional-hooks-dir=./ \
--onefile --windowed pyvls.pyw
