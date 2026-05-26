# SubMeet Classification Service

Service ini menangani klasifikasi topik abstrak/judul paper menggunakan model machine learning, lalu mengirim hasil prediksi ke webhook yang ditentukan.

## Fitur

- REST API berbasis FastAPI.
- Prediksi topik paper secara asynchronous melalui background task.
- Integrasi preprocessing teks, model CNN, dan threshold berbasis sigma.
- Endpoint mock webhook untuk kebutuhan testing lokal.

## Struktur Proyek

```text
main.py
api/
  routers/
    topic_detection_router.py
config/
domain/
  models/
  services/
infrastructure/
  ml/
  model_files/
  preprocessing/
schemas/
```

## Requirements

- Python 3.10 atau yang kompatibel dengan dependency yang dipakai.
- Virtual environment aktif.
- File model tersedia di folder `infrastructure/model_files/`.

## Instalasi

1. Buat dan aktifkan virtual environment.
2. Install dependency:

```bash
pip install -r requirements.txt
```

## Konfigurasi Environment

Buat file `.env` di root project. Contoh:

```env
APP_NAME=SubMeet Classifier Service
PORT= 8001

CNN_MODEL_PATH=infrastructure/model_files/ (model CNN keras)
SIGMAS_PATH=infrastructure/model_files/ (sigma.json)
VOCAB_PATH=infrastructure/model_files/ (word_to_idx.json)
```

### Keterangan

- `CNN_MODEL_PATH` menunjuk ke file model Keras CNN.
- `SIGMAS_PATH` menunjuk ke file threshold/sigma untuk klasifikasi.
- `VOCAB_PATH` menunjuk ke file vocabulary untuk preprocessing.
- Path boleh diganti jika model baru dipindahkan, selama file-nya tetap valid.

## Menjalankan Aplikasi

Jalankan server dengan Uvicorn, sesuai dengan portnya:

```bash
uvicorn main:app --reload --port 8001
```

Jika ingin memakai port dari `.env`, jalankan dengan port yang sama seperti nilai `PORT`.

## Endpoint

### `POST /api/detect`

Menerima request klasifikasi topik dan memprosesnya di background.

#### Request Body

```json
{
  "paper_sub_id": 123,
  "title": "Judul paper",
  "abstract": "Isi abstrak paper",
  "webhook_url": "https://example.com/webhook"
}
```

#### Response

```json
{
  "status": "accepted",
  "job_id": "uuid-baru",
  "message": "Permintaan klasifikasi sedang diproses di latar belakang."
}
```

### `POST /api/mock-laravel-webhook`

Endpoint dummy untuk testing payload hasil prediksi sebelum dihubungkan ke sistem utama.

## Alur Kerja

1. Request masuk ke `POST /api/detect`.
2. Judul dan abstrak digabung.
3. Teks dibersihkan dan diubah menjadi sequence.
4. Model CNN melakukan prediksi.
5. Threshold sigma dibandingkan dengan confidence score.
6. Hasil prediksi dikirim ke `webhook_url`.

## Catatan Implementasi

- Hyperparameter seperti `alpha` tetap di-hardcode sesuai kebutuhan eksperimen.
- Path file model sebaiknya dikelola lewat `.env` agar pergantian model tidak perlu ubah kode.
- File model utama tetap disimpan di `infrastructure/model_files/`.

## Testing Lokal

Untuk mencoba alur penuh tanpa sistem eksternal, arahkan `webhook_url` ke endpoint mock webhook di aplikasi ini.