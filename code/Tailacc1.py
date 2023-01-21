import matplotlib.pyplot as plt

ten_best = ['/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_204.jpg'
, '/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_39519.jpg'
 ,'/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_6476.jpg'
, '/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_18053.jpg'
 ,'/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_29734.jpg'
, '/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_23233.jpg'
 ,'/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_17137.jpg'
, '/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_7767.jpg'
 ,'/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_6838.jpg'
 ,'/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_34753.jpg']
ten_worst = ['/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_21936.jpg',
 '/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_11673.jpg'
, '/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_15313.jpg'
 ,'/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_14862.jpg'
 ,'/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_38807.jpg'
 ,'/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_13497.jpg'
, '/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_2595.jpg'
 ,'/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_39863.jpg'
, '/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_6893.jpg'
 ,'/itf-fi-ml/shared/IN5400/2022_mandatory1/train-jpg/train_19420.jpg']

fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 5
Image1 = plt.imread(ten_best[0])
fig.add_subplot(rows, columns, 1)
plt.imshow(Image1)
plt.axis('off')
plt.title("First")
Image2 = plt.imread(ten_best[1])
fig.add_subplot(rows, columns, 2)
plt.imshow(Image2)
plt.axis('off')
plt.title("Second")
Image3 = plt.imread(ten_best[2])
fig.add_subplot(rows, columns, 3)
plt.imshow(Image3)
plt.axis('off')
plt.title("Third")
Image4 = plt.imread(ten_best[3])
fig.add_subplot(rows, columns, 4)
plt.imshow(Image4)
plt.axis('off')
plt.title("Fourth")
Image5 = plt.imread(ten_best[4])
fig.add_subplot(rows, columns, 5)
plt.imshow(Image5)
plt.axis('off')
plt.title("Fifth")
Image6 = plt.imread(ten_best[5])
fig.add_subplot(rows, columns, 6)
plt.imshow(Image6)
plt.axis('off')
plt.title("Sixth")
Image7 = plt.imread(ten_best[6])
fig.add_subplot(rows, columns, 7)
plt.imshow(Image7)
plt.axis('off')
plt.title("Seventh")
Image8 = plt.imread(ten_best[7])
fig.add_subplot(rows, columns, 8)
plt.imshow(Image8)
plt.axis('off')
plt.title("Eight")
Image9 = plt.imread(ten_best[8])
fig.add_subplot(rows, columns, 9)
plt.imshow(Image9)
plt.axis('off')
plt.title("Nine")
Image10 = plt.imread(ten_best[9])
fig.add_subplot(rows, columns, 10)
plt.imshow(Image10)
plt.axis('off')
plt.title("Tenth")

plt.savefig("10best.pdf")
plt.close()

fig_ = plt.figure(figsize=(10, 7))
rows = 2
columns = 5
Image1_ = plt.imread(ten_worst[0])
fig_.add_subplot(rows, columns, 1)
plt.imshow(Image1_)
plt.axis('off')
plt.title("First")
Image2_ = plt.imread(ten_worst[1])
fig_.add_subplot(rows, columns, 2)
plt.imshow(Image2_)
plt.axis('off')
plt.title("Second")
Image3_ = plt.imread(ten_worst[2])
fig_.add_subplot(rows, columns, 3)
plt.imshow(Image3_)
plt.axis('off')
plt.title("Third")
Image4_ = plt.imread(ten_worst[3])
fig_.add_subplot(rows, columns, 4)
plt.imshow(Image4_)
plt.axis('off')
plt.title("Fourth")
Image5_ = plt.imread(ten_worst[4])
fig_.add_subplot(rows, columns, 5)
plt.imshow(Image5_)
plt.axis('off')
plt.title("Fifth")
Image6_ = plt.imread(ten_worst[5])
fig_.add_subplot(rows, columns, 6)
plt.imshow(Image6_)
plt.axis('off')
plt.title("Sixth")
Image7_ = plt.imread(ten_worst[6])
fig_.add_subplot(rows, columns, 7)
plt.imshow(Image7_)
plt.axis('off')
plt.title("Seventh")
Image8_ = plt.imread(ten_worst[7])
fig_.add_subplot(rows, columns, 8)
plt.imshow(Image8_)
plt.axis('off')
plt.title("Eight")
Image9_ = plt.imread(ten_worst[8])
fig_.add_subplot(rows, columns, 9)
plt.imshow(Image9_)
plt.axis('off')
plt.title("Nine")
Image10_ = plt.imread(ten_worst[9])
fig_.add_subplot(rows, columns, 10)
plt.imshow(Image10_)
plt.axis('off')
plt.title("Tenth")

plt.savefig("10worst.pdf")
plt.close()
