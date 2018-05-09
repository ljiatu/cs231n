from dataset import IMDbFacialDataset
import matplotlib.pyplot as plt


def main():
    dataset = IMDbFacialDataset('imdb_crop')
    image = dataset[0]
    print(0, image.shape)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    main()
