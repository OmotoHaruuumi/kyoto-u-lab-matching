# 京都大学 研究室マッチングシステム — TODO / 進捗管理

> 最終更新: 2026-06-08
> このファイルはプロジェクトの進捗とこれからやることを管理するためのものです。

## プロジェクト概要

京都大学の研究室を集約・ベクトル化し、ハイブリッド検索（ベクトル + キーワード、Weighted RRF）で
マッチングするシステム。想定ユーザーは産学連携を探す企業・共同研究を探す研究室・進学先を探す学生。

**技術スタック**: Next.js 16 / FastAPI + pgvector / PostgreSQL 16 / Redis 7 / Playwright + Gemini / Docker Compose

---

## ✅ 完成している機能（現状）

### インフラ・基盤
- [x] Docker Compose 構成（frontend / search_api / embedding_api / postgres / redis / migrate / crawler）
- [x] PostgreSQL + pgvector、マルチステージビルドの Dockerfile
- [x] Alembic マイグレーション（初期スキーマ / HNSW index / keyword tiers の3世代）
- [x] `.env.example` と環境変数リファレンス（README に整備済み）

### データモデル（`backend/shared/models.py`）
- [x] `labs` / `professors` / `research_themes` / `embedding_chunks` / `data_sources`
- [x] keywords を primary / secondary の2階層に分割
- [x] embedding は HNSW index（cosine ops、m=16 / ef_construction=64）

### 検索 API（`backend/search_api/main.py`）
- [x] `/health`、`/api/v1/categories`、`/api/v1/labs`（faculty/department フィルタ）、`/api/v1/search`
- [x] Weighted RRF によるハイブリッド検索（ベクトル 0.7 / キーワード 0.3）
- [x] バイリンガル クエリリライト（Gemini で日英キーワード拡張）
- [x] chunk-type 別の重み付け（lab_description を research_theme より優先）
- [x] Redis キャッシュ、ページネーション対応

### 埋め込み API（`backend/embedding_api/main.py`）
- [x] Gemini `text-embedding-004`（768次元固定）
- [x] `MOCK_EMBEDDING=true` でAPIキー不要のモック動作

### クローラー（`crawler/`）
- [x] Playwright によるマルチページクロール
- [x] Gemini `gemini-2.5-flash` による構造化抽出（`extractor.py`）
- [x] DB保存 + 埋め込み生成（`loader.py`）
- [x] `categories.json` による faculty/department のバリデーション・正規化
- [x] CSV（`urls.csv`、59行）から faculty/department をパイプライン全体に伝播
- [x] URL 収集スクリプト（`collect_urls.py`、`index_urls.txt` 55件）

### フロントエンド（`frontend/src/app/`）
- [x] 検索タブ + 研究室一覧タブ（タブ切替）
- [x] 検索結果のSkeletonローディング、マッチchunk表示
- [x] 研究室詳細モーダル（`DetailModal`）
- [x] faculty/department によるフィルタリング + ページネーション

### テスト（`tests/`）
- [x] pytest スイート（search_api / embedding_api、計30テスト関数）

---

## 🚧 これからやること（TODO）

### クロールの現状（2026-06-08 本番DBで確認済み）
- ✅ **情報学研究科 = 57研究室、クロール完了**。`data_sources` は57件すべて `done`（失敗ゼロ）。
      `urls.csv`（ヘッダ除き57件）と完全一致＝取りこぼし無し。
  - 内訳：知能情報学13 / 社会情報学13 / 通信情報システム9 / システム科学9 / 数理工学6 / 先端数理科学4 / データ科学3
- ✅ 本番はさくらのクラウド上で **Docker 5コンテナが2ヶ月間稼働中**（IP `49.212.128.36`、host `os3-318-48532`）。
      データ実体は `kyoto_lab_db`（pgvector/pg16）コンテナ内。
- ❌ **情報学研究科以外（工学・エネルギー科学等）は未クロール**。`categories.json` には定義済みだが
      `urls.csv` には未登録。
- 📝 **コース/研究科の分類は「半自動＋手動仕上げ」**。仕組み上は CSV指定(faculty_override) > AI抽出 で
      `categories.json` 正規化（[loader.py:130](crawler/loader.py#L130)）だが、本番57件が分類null=0でキレイなのは
      最後に**手作業で1件ずつ修正したため**（本人の記憶）。
      → **拡大時の含意**: コース分類は自動だけでは今の品質に届かない。クロール後に手動レビュー/修正の工程が要る。
      （現 `urls.csv` は `url,source_page` の2列のみで faculty/department 列が無い点も要整理）

### P1: データ整備（最優先 — システムの価値の源泉）
- [x] 本番DBの現状確認 → 情報学研究科57件すべてクロール成功を確認
- [x] **本番DBのバックアップ取得**（2026-06-08 完了）
      - [x] サーバー上で `pg_dump -Fc` → `~/kyoto_lab_db_20260608.dump` 作成
      - [x] Mac へ scp ダウンロード → `backups/kyoto_lab_db_20260608.dump`（2.5MB、カスタム形式・検証OK）
      - [x] `.gitignore` の `*.dump` でGit対象外を確認済み
      - [ ] （任意）定期バックアップの自動化・別ストレージへの退避
- [x] **クロール結果の品質確認**（2026-06-08 完了）
      - ✅ 検索の根幹は良好: description / lab_url / keywords_primary は **57/57 全件あり・内容も的確**
      - ⚠️ name_en（英名）が **14/57 欠損**
      - ⚠️ **教授抽出が系統的に弱い**（致命的ではない＝二次的メタデータ）:
        - 教授数の分布: 0人=3, 1人=23, 2人=13, 3人=8, 4=1, 5=5, 6=3, 7=1 → **46%が1人以下**＝主担当しか拾えていない
        - **email は全件空**（一切抽出できていない）
        - 氏名が姓のみのケースが多い（フルネーム取得が不安定）
        - title は概ね取得できている

### P1.4: 再クロールの自動化（コース分類の手作業を撤廃）— 2026-06-08 実装完了
毎年の再クロールでコースを手修正する運用を撤廃。コースは「URLの出所」で決定的に固定する。
- [x] **① `urls.csv` を4列化**（`url, source_page, faculty, department`）。既存57URLに index_urls.txt から
      faculty/department を決定的に付与。今後 `collect_urls.py` 出力をそのまま使う運用に（READMEに明記）
- [x] **② `crawler/overrides.csv` 導入**。手動上書きをDB直編集ではなくファイルに永続化し再クロールで自動再適用。
      優先順位は overrides.csv > urls.csv > AI抽出（[loader.py](crawler/loader.py) で実装）
- [x] **③ 再クロールの上書き更新**。`CRAWL_FORCE_REFRESH=true` で既存レコードを削除→入れ直す upsert。
      既定はスキップ（冪等）。[main.py](crawler/main.py) / [loader.py](crawler/loader.py) / docker-compose に実装
- 補足: `urls.csv` は58URLだが本番DBは57ラボ（数理工学コースで1件差）。再クロール時に要確認
- [ ] 次回再クロールで実際に動作確認（本番反映前に検証環境推奨）

### P1.5: データ品質の改善（任意・優先度中。検索の根幹は既に機能するため急がない）
- [ ] name_en 欠損14件の補完（DB上の該当labを再抽出 or 手動補完）
- [ ] `crawler/extractor.py` の教授抽出強化（メンバー一覧ページの巡回・email正規表現・フルネーム取得）
- [ ] 既存57件の教授情報を再抽出して上書きするか判断（産学連携の連絡先ユースケース次第）
- [ ] **対象研究科の拡大**：情報学研究科以外のURL収集 → `urls.csv` 追加 → クロール
      （`collect_urls.py` と `categories.json` を活用）
- [ ] クロールの冪等性・差分更新の確認（`data_sources.content_hash` を活用した再クロール）
- [ ] クロール失敗時のリトライ／エラーハンドリングの検証（`status=failed` の扱い）

### P2: 検索品質の向上
- [ ] 実データでの検索精度評価（RRF重み・RRF_K・chunk重みのチューニング）
- [ ] professor（教授名）を検索対象に含めるか検討（現状 search では未活用）
- [ ] 検索のベンチマーク／評価用のクエリセット作成
- [ ] 類似研究室レコメンド（lab間のベクトル近接度）— 「研究室同士の近さ」のユースケース

### P3: フロントエンド / UX
- [ ] 研究室一覧の検索ボックス内フィルタ（キーワードでの絞り込み）
- [ ] 詳細モーダルに professors / research_themes を表示
- [ ] レスポンシブ・モバイル対応の確認
- [ ] エラー状態・空状態のUI改善
- [ ] i18n（英語UI）— 企業・海外向けユースケース

### P4: 運用・本番化
- [ ] 本番デプロイ構成（ホスティング先の決定、シークレット管理）
- [ ] 定期クロールの自動化（cron / スケジューラ）
- [ ] ログ・モニタリング・ヘルスチェックの運用整備
- [ ] CI（テスト自動実行）の追加
- [ ] レート制限・APIキー管理（Gemini のコスト管理）

### P5: ドキュメント
- [ ] `CLAUDE.md` の作成（開発ガイド・アーキテクチャ要約）
- [ ] クローラーの運用手順ドキュメント
- [ ] API ドキュメント（FastAPI の OpenAPI を活用）

---

## ❓ 確認が必要な未確定事項（要・本人判断）

- [ ] **本番環境はどこか**（ホスティング先・URL）、現在も稼働しているか
- [ ] **本番DBへのアクセス手段**（別PC / クラウド）、バックアップは取れるか
- [ ] 情報学研究科のクロール成功件数（58件中いくつ入っているか）
- [ ] `GEMINI_API_KEY` は手元にあるか（追加クロール・本番検索に必須）
- [ ] 次に拡大する研究科の優先順位（工学／エネルギー科学／全学 など）

---

## メモ / 次に着手するなら

1. **本番DBの状態確認とバックアップ**（別PC依存で消失リスクがあるため最優先）
2. ローカルで再現したい場合：本番DBをダンプ → ローカルの `postgres_data` にリストア
   （または `MOCK_EMBEDDING=true` でサービス起動 → seed で動作確認のみ）
3. データ拡大：情報学研究科以外のURLを `collect_urls.py` で収集 → クロール → 検索精度チューニング
