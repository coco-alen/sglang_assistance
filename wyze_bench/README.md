## client_send_wyze_data_1fps.py

Used to simulate sending multiple image requests to the server at 1fps, using the original image

> Note：This usually results in too many large images, and requires turning on *frame_pruning* on the server for testing.

Send the real data of wyze bench,

```bash
python benchmark/wyze_bench/client_send_wyze_data_1fps.py baby 64 --port 30457
```

'baby' for category. Will automatically fetch the video under --data_path/category.

'64' for concurrency.  Number of concurrent API calls



## client_send_dummy_data.py

Used to simulate sending noise image to the server to test the throughput.

> Note：You can specify the number of images to request at a time, so it can be used to simulate the effect after frame_pruning. It is suitable for use when there is no *wyze_data* or when you want to isolate part of the *frame_pruning* time.

```bash
python wyze_bench/client_send_dummy_data.py --min-images-per 1 --max-images-per 1 --max-concurrency 64 --base-url http://localhost:23333/v1 --num-requests 200
```



## client_send_dummy_data_to_2server.py

Used to simulate sending requests to two servers at the same time to test the speed improvement brought by scaling.

```bash
python wyze_bench/client_send_dummy_data_to_2server.py --min-images-per 5 --max-images-per 30 --max-concurrency 64 --base-url http://localhost:30
456/v1 --base-url-2 http://localhost:30457/v1 --num-requests 300
```

