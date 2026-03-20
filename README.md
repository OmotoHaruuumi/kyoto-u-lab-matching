# 京都大学 研究室マッチングシステム

企業担当者が自然言語で研究課題を入力し、関連する京都大学の研究室をAIで検索できるWebシステムです。

## 技術スタック

| レイヤー | 技術 |
|---|---|
| フロントエンド | Next.js 16 (App Router, TypeScript, Tailwind CSS v4) |
| 検索 API | FastAPI + pgvector (Hybrid Search + Weighted RRF) |
| 埋め込み API | FastAPI + Google Gemini `text-embedding-004` |
| データベース | PostgreSQL 16 + pgvector拡張 |
| キャッシュ | Redis 7 |
| クローラー | Playwright + Google Gemini `gemini-2.5-flash` |
| コンテナ | Docker Compose (マルチステージビルド) |

### 検索アーキテクチャ

```
ユーザークエリ
    │
    ├─► ベクトル検索 (pgvector cosine distance, 重み 0.7)
    │       └── Embedding API → Gemini text-embedding-004
    │
    └─► キーワード検索 (トークン分割 OR-ILIKE, 重み 0.3)
            │
            └── Weighted RRF (k=30) で統合 → 研究室ランキング
```

---

## クイックスタート

### 前提条件

- Docker Desktop または Docker Engine + Docker Compose v2
- Git

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd kyoto-u-lab-matching
```

### 2. 環境変数の設定

```bash
cp .env.example .env
```

`.env` を開き、以下の値を設定します。

| 変数 | 説明 |
|---|---|
| `POSTGRES_PASSWORD` | PostgreSQL のパスワード |
| `DATABASE_URL` | SQLAlchemy 接続文字列 |
| `GEMINI_API_KEY` | Google AI Studio で取得したAPIキー |
| `MOCK_EMBEDDING` | `true` にすると Gemini を呼ばずにモック埋め込みを使用 |

> **ローカル開発では `MOCK_EMBEDDING=true` を推奨します。**
> Gemini API キーなしで全機能をテストできます。

### 3. 起動

```bash
docker compose up --build
```

初回起動には数分かかります（イメージのビルド）。

### 4. データ投入（初回のみ）

別ターミナルで：

```bash
# マイグレーション
docker compose exec search_api python -m alembic -c /app/db/alembic.ini upgrade head

# ダミーデータ投入（5件の研究室）
docker compose exec search_api python db/seed.py
```

実際の京都大学研究室データを収集する場合はクローラーを実行します：

```bash
# GEMINI_API_KEY が必要
docker compose --profile crawler up crawler
```

### 5. アクセス

| サービス | URL |
|---|---|
| フロントエンド | http://localhost:3000 |
| 検索 API (Swagger UI) | http://localhost:8000/docs |
| 埋め込み API (Swagger UI) | http://localhost:8001/docs |

---

## API 仕様

### 検索 API (`http://localhost:8000`)

#### `GET /api/v1/search`

研究室のハイブリッド検索を実行します。

**リクエスト**

| パラメータ | 型 | 必須 | 説明 |
|---|---|---|---|
| `q` | string | ✅ | 検索クエリ（自然言語） |
| `limit` | integer | ✗ | 取得件数 (1–50, デフォルト 10) |

**レスポンス例**

```json
{
  "query": "機械学習を使った医療診断",
  "results": [
    {
      "lab_id": 1,
      "name": "自然言語処理研究室",
      "name_en": "Natural Language Processing Lab",
      "department": "情報学研究科",
      "faculty": "工学部",
      "lab_url": "https://nlp.ist.i.kyoto-u.ac.jp/",
      "description": "...",
      "keywords": ["NLP", "機械学習", "深層学習"],
      "matched_chunks": [
        {
          "chunk_text": "...",
          "source_type": "lab_description",
          "combined_score": 0.021
        }
      ],
      "total_score": 0.042
    }
  ]
}
```

#### `GET /health`

DB・Redis の接続状態を返します。

```json
{ "status": "ok", "db": "ok", "redis": "ok" }
```

---

### 埋め込み API (`http://localhost:8001`)

#### `POST /api/v1/embed`

テキストの埋め込みベクトルを生成します。

**リクエスト**

```json
{
  "texts": ["機械学習を用いた医療診断の研究"],
  "task_type": "RETRIEVAL_DOCUMENT"
}
```

**レスポンス**

```json
{
  "embeddings": [[0.012, -0.034, ...]],
  "model": "models/text-embedding-004",
  "dim": 768,
  "mock": false
}
```

`task_type` は `RETRIEVAL_DOCUMENT`（文書登録時）または `RETRIEVAL_QUERY`（検索時）を指定します。

---

## ディレクトリ構成

```
kyoto-u-lab-matching/
├── frontend/                 # Next.js フロントエンド
│   ├── src/app/
│   │   ├── page.tsx          # 検索UI（Skeleton ローディング付き）
│   │   ├── actions.ts        # Server Action（search_api 呼び出し）
│   │   └── layout.tsx
│   └── Dockerfile            # マルチステージビルド
│
├── backend/
│   ├── search_api/           # 検索 API (FastAPI)
│   │   ├── main.py           # Weighted RRF ハイブリッド検索
│   │   └── Dockerfile        # マルチステージビルド
│   ├── embedding_api/        # 埋め込み API (FastAPI + google-genai)
│   │   ├── main.py
│   │   └── Dockerfile
│   └── shared/
│       ├── database.py       # 非同期 SQLAlchemy エンジン
│       └── models.py         # ORM モデル（labs, embedding_chunks など）
│
├── crawler/                  # Playwright + Gemini クローラー
│   ├── main.py               # エントリーポイント
│   ├── extractor.py          # Gemini による構造化抽出
│   └── loader.py             # DB 保存 + 埋め込み生成
│
├── db/
│   ├── alembic/              # マイグレーション
│   └── seed.py               # ダミーデータ投入スクリプト
│
├── tests/                    # Pytest テストスイート
│   ├── test_embedding_api.py
│   └── test_search_api.py
│
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 開発用コマンド

```bash
# 全サービス起動
docker compose up

# ヘルスチェック確認
curl http://localhost:8000/health | python3 -m json.tool
curl http://localhost:8001/health | python3 -m json.tool

# 検索テスト
curl -G http://localhost:8000/api/v1/search \
  --data-urlencode "q=機械学習" -d "limit=3" | python3 -m json.tool

# テスト実行（仮想環境で）
pip install pytest pytest-asyncio httpx
pytest tests/ -v

# クローラー実行（Gemini API キー必須）
docker compose --profile crawler up crawler

# 全サービス停止・ボリューム削除
docker compose down -v
```

---

## 環境変数リファレンス

| 変数 | デフォルト | 説明 |
|---|---|---|
| `POSTGRES_USER` | `labmatch` | PostgreSQL ユーザー名 |
| `POSTGRES_PASSWORD` | — | PostgreSQL パスワード（必須） |
| `POSTGRES_DB` | `labmatch_db` | データベース名 |
| `DATABASE_URL` | — | SQLAlchemy 接続文字列（必須） |
| `REDIS_URL` | `redis://redis:6379/0` | Redis 接続文字列 |
| `GEMINI_API_KEY` | — | Google Gemini API キー |
| `MOCK_EMBEDDING` | `false` | `true` でモック埋め込みを使用 |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | 使用する埋め込みモデル |
| `EMBEDDING_API_URL` | `http://embedding_api:8001` | search_api → embedding_api の内部URL |
| `SEARCH_API_URL` | `http://search_api:8000` | frontend → search_api の内部URL |
| `RRF_K` | `30` | RRF定数（小さいほどランク差が強調） |
| `RRF_VECTOR_WEIGHT` | `0.7` | ベクトル検索スコアの重み |
| `RRF_KEYWORD_WEIGHT` | `0.3` | キーワード検索スコアの重み |
| `SEARCH_CANDIDATE_LIMIT` | `100` | 各検索モードの候補数 |
| `SQLALCHEMY_ECHO` | `false` | `true` で SQL ログを出力 |
