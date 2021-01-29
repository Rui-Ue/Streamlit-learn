import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
import time
import matplotlib.pyplot as plt
import glob

from PIL import Image

import base64




# タイトル。日本語もいける。
st.title("自販機画像認識アプリ")
st.header('ヘッダー')
st.subheader('サブヘッダー')
# これらは後述の st.markdown() での #, ##, ### と同じらしい。



# 文を書く。
st.write("アップロードされた自販機の画像から、自動でカテゴリと価格を抽出します。")
st.write("aaaa")
# write を複数回呼び出すと、改行が入る。




# このデコレータなんだろう？データの中身が更新されたら即時反映されるためとか？？
# @st.cache
# def load_data():
#     return df

# 日本語もいけた。
df = pd.DataFrame({
    "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "自販機ID": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    "カテゴリ": ["お茶", "COFFEE", "BLACK_TEA", "NS_TEA", "WATER",
                 "SSD", "BLACK_TEA", "COFFEE", "FRUIT_JUUCE", "WATER"],
    "Price": [110, 90, 100, 120, 140, 100, 100, 110, 120, 100]
    # 価格の部分は、OCRがまだだが仮で作りました、で良い。
                
})

# 抽出した飲料カテゴリと価格の DF はこれで表示すればOK。
st.dataframe(df)




st.write("上記が出力データです。\n\n 改行入れた。")
# こうすれば空行を入れることもできる。md の <br> 的な。




md_str = \
"""ここからマークダウン
md の一行目

# 第1段落
## 第2段落
### 第3段落

いろいろためす。
- 箇条書きも
- できる
  - ネストもできる。
- 便利だな。

[リンク](https://github.com/Rui-Ue)もはれている。

画像もいけるか？
![img](file:///Users/rui/learn_tech/streamlit_learn/data/zihanki.jpg)
![img](/Users/rui/learn_tech/streamlit_learn/data/zihanki.jpg)
![img](data/zihanki.jpg)
ok -> [zihanki](https://user-images.githubusercontent.com/55879719/106141952-30765100-61b4-11eb-8d42-cbddef348711.jpg)

ローカルだとだめで、リモート github のやつはいけた。
"""
# print(md_str)

st.markdown(md_str)




code_str = \
"""
import numpy as np
import pandas as pd

print('Hello')
"""
st.code(code_str, language="python")
# st.markdown() で ```python コード``` とやっても同じように表示できるらしい。
# となると f"""マークダウンでたくさん""" を st.markdown() するのが最強では？




detection_flag = st.button("検出実行")  # ボタンが押されると True が返ってくる。

if detection_flag:
    # print("実行しました")  # これだとこの python 実行環境の標準出力に出ちゃう。
    st.write("実行しました")
# ボタンを押すとこの「実行しました」が表示されるってことは、if 文の判定が常に高速で回り続けてるのか？
# そのあたりの仕組みはわからない。
# ↓
# notion へ。
# > 起動時やボタンを押すなどで、プログラムを先頭から（同一プロセスで）再実行します。 ※ @st.cacheを関数定義につけると、同一引数の呼び出しでキャッシュを使うようになります。

# このボタンはそのまま使えそう。
# アップロードした状態で、ボタンを押すとようやく、実行される。
# もとに戻すのはどうするかわからんが。




check = st.checkbox("csvにもエクスポートする")
if check:
    st.write("結果画像表示するだけでなくエクスポートもします。")




mode = st.radio(
    "検出モードを選んでください",
    ("カテゴリ", "価格")
)

if mode == "カテゴリ":
    st.write("カテゴリを押してますね")
elif mode == "価格":
    st.write("価格を押してますね。")
# このラジオボタンも、別の選択肢を選ぶたびに、if 文の判定が実行されなおされてる。
# どういう仕組みだろう。




trg_img = st.selectbox(
    "検出結果をみたい画像を選んでください",
    ("img_miu001", "img_miu002", "uos_img003", "uos_img004")
)
st.write(f"画像{trg_img}を選んでいますね")

# これも、検出結果画像を 1 枚 1 枚確認するのに使える。
# デモで、このドロップダウンを切り替えると結果画像が切り替わる部分を見せたい。




# 地図上に散布図プロット

map_df = pd.DataFrame({
    "lat": [35.462291, 35.457767, 35.334138],
    "lon": [139.527413, 139.535789, 139.617897]
})
# 家, まき中, 大学

st.map(map_df)

# 新規設置場所の立案の支援システムに使える？




# プログレスバー、かっこいい

if st.button("検出の一括実行"):
    cont = st.empty()  # コンテナ?を作ってるらしいが未理解。ドキュメントは https://docs.streamlit.io/en/stable/api.html?highlight=empty#streamlit.empty
    bar = st.progress(0) # プログレスバー
    for i in range(100):
        cont.text(f"検出実行中 {i+1}/{100} {(i+1)/100*100}%")  # バーが進むと同時にコンテナ内のテキストを更新してそれっぽくする。
        bar.progress(i + 1) # バー自体はこれで進む
        time.sleep(0.2)

# このプログレスバーは、画像 70 枚を突っ込んだときに全体の何%の画像の検出が終わってるか、で見せたい。
# ファイルダウンロードの時もこのバー出す？




# matplotlib.pyplot のグラフを表示

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1) # キャンパスを1行2列に区切った1つ目に描く
ax1.pie(
    np.array([1, 2, 3, 2, 1]),
    labels=["a", "b", "c", "d", "e"]
)
ax2 = fig.add_subplot(1,2,2)
ax2.bar(
    list(range(1, df["カテゴリ"].nunique()+1)),
    list(df["カテゴリ"].value_counts()),
    tick_label=list(df["カテゴリ"].value_counts().index)
)
plt.xticks(rotation=90)
st.pyplot(fig)

# matplotlib なのでそのままでは日本語表示ができない。






# 公式 https://docs.streamlit.io/en/stable/api.html?highlight=area_chart#display-media より、
# 画像の表示方法
# keras-yolo3 と同じ PIL モジュールで可能！

image = Image.open("./data/zihanki.jpg")
st.image(
    image,
    caption="vending machine",
    use_column_width=True  # 画像の幅を列の幅(表示上の幅？)にする。
    # この use_column をつけないと、ブラウザを画面右半分で開いてる時に、なんかうまく見えてる範囲におさまらなかった。
    # この引数つけたら、おさまるようになった。
)




# ユーザーからのテキスト入力, 数値入力

in_txt = st.text_input("適当な文字列を入れてね", value="デフォルト値")
st.write(f"あなたは {in_txt} と入力しましたね。")

in_num = st.number_input("0~10の数字を入れてね", min_value=0, max_value=10)
st.write(f"あなたは {in_num} と入力しましたね。")




# ファイルのアップロード
# https://docs.streamlit.io/en/stable/api.html?highlight=area_chart#streamlit.file_uploader

uploaded_files = st.file_uploader(
    label="アップロード（複数可・合計200MBまで）",
    type="jpg",  # jpg がアップロードされると指定
    accept_multiple_files=True  # 複数ファイルのアップロードを許可
)

# if len(uploaded_files) > 1:
#     st.write(f"{len(uploaded_files)}, {type(uploaded_files[0])}, {type(uploaded_files[1])}")
#     st.write(uploaded_files[0].name, uploaded_files[1].name, uploaded_files[2].name)
# 3, <class 'streamlit.uploaded_file_manager.UploadedFile'>, <class 'streamlit.uploaded_file_manager.UploadedFile'>
# なるほど、こういう UploadedFile クラスのオブジェクトとして読み込まれるのか画像が。
# これ、どうやれば PIL.Image オブジェクトに変換できるんだろう。
# -> notion
# > The UploadedFile class is a subclass of BytesIO, and therefore it is “file-like”. This means you can pass them anywhere where a file is expected.
# この UpleadedFile クラスは BytesIO クラスの子らしい。
# https://techacademy.jp/magazine/19185 より、
# > BytesIOとは、メモリ上でバイナリデータを扱うための機能です。Python の標準ライブラリ io に含まれています。バイナリデータとは主に画像や音声などのデータのことです。コンピューターで扱うデータは全てバイナリデータなのですが、テキストデータと対比して用いられます。

# 諸々の仕組みは notion 参照。

if len(uploaded_files) > 0:
    trg_img = st.selectbox(
        "見たいアップロード画像を選んでください",
        [f.name for f in uploaded_files]
    )
    st.write(f"画像{trg_img}を選んでいますね")
    u_img = Image.open([f for f in uploaded_files if f.name==trg_img][0])
    st.image(u_img)

# シンプルに、アップロードされたファイルを web サーバ (mac, ubuntu ローカル) に一回保存して、
# そんでそのローカルファイルを OPEN する、とした方が、自然で楽かもしれない。
# なんていうか、小分けにしてアップロードする人もいるだろうし。
# その方式でやってみる↓

# if len(uploaded_files) > 0:
#     for f in uploaded_files:
#         Image.open(f).save(f"./data/uploaded_files/{f.name}")

# ↑当たり前だが、これだと、ファイルアップロードされた状態で何らかのボタン操作するたびに、上書き save される。
# パフォーマンスが落ちて画面がガタガタになるだろうな。
# こういうときに、notion 記述の @st.cache とかを使うのかなあ。
# どうにかして「ファイルアップロード直後の時だけ open(f).save() が動く」というようにしたいが。。。
# 強引にやるとしたら、glob.glob で保存済みアップロード画像を毎回チェックすれば良い。それで実装してみる↓

if len(uploaded_files) > 0:
    already_uploaded_files = [f.split("/")[-1] for f in glob.glob("./data/uploaded_files/*")]
    for f in uploaded_files:
        if f.name not in already_uploaded_files:
            Image.open(f).save(f"./data/uploaded_files/{f.name}")

# ひとまずこれで良いや。
# 本当は、「同じ内容の画像は必ず同じファイル名, ファイル名が ID になっている」という前提煮立ってしまってるので、そこがダメ。
# その部分は、本来はちゃんと対応しないといけない。




# ファイルのダウンロード。notion 参照

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    # href = f'<a href="data:file/csv;base64,{b64}">検出結果をダウンロード</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="detection_result.csv">検出結果をダウンロード</a>'
    # これでダウンロード時のファイル名も指定できる！！
    return href

st.markdown(get_table_download_link(df), unsafe_allow_html=True)




"""
自販機 自動販売機
抽出 実行 検出 画像認識
アップロード ダウンロード データ ファイル 結果

自販機画像をアップロード

アップロードした画像を選んでください
表示したい画像を選択
画面

リセット 結果のリセット 結果をリセット

一括 実行

撮影画像

検出結果画像

検出されたドリンク飲料カテゴリデータ

検出を実行（一括）
検出を一括実行
で

"""