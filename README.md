# RAG_local
このソフトウェアはプライベート用PCにRAGを構築することができます。  
HDDに格納したデータを学習し、chatbot形式で回答します。  
動作確認済OS:WIndows11

## リポジトリをgit cloneする
まず、対象のリポジトリをローカル環境にクローンする必要があります。ターミナルまたはコマンドプロンプトを開き、以下のコマンドを実行してください。

```
git clone <リポジトリのURL>
```
## Python3をインストールする
次に、Python3をインストールする必要があります。まだインストールされていない場合は、公式サイトからインストーラをダウンロードし、インストールを行ってください。

Python仮想環境を作成する
プロジェクトごとに独立した環境を作成することが推奨されています。ターミナルまたはコマンドプロンプトで、クローンしたリポジトリのディレクトリに移動し、以下のコマンドを実行して仮想環境を作成してください。

```
python3 -m venv env
```
仮想環境を有効化する
作成した仮想環境を有効化します。

Windows:

```
.\venv\Scripts\activate
```

requirement.txtを参考にライブラリをインストールする
プロジェクトで使用しているライブラリをインストールします。通常、プロジェクトのルートディレクトリに requirements.txt ファイルが存在し、そこに必要なライブラリが記載されています。以下のコマンドを実行してください。
```
pip install -r requirements.txt
```
.envにOPEN AI keyを入力する
プロジェクトが OpenAI の API キーを使用している場合は、.env ファイルを作成し、そこに API キーを設定する必要があります。.env ファイルを作成し、以下の形式で API キーを設定してください。

```
OPENAI_API_KEY=<your_openai_api_key>
```
Streamlitを起動する
最後に、Streamlitアプリケーションを起動します。ターミナルまたはコマンドプロンプトで以下のコマンドを実行してください。

```
streamlit run app.py
```
このコマンドを実行すると、ローカルサーバーが立ち上がり、Streamlitアプリケーションが表示されます。
