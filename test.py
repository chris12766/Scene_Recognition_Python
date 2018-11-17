'''
#Display images
x,y = train_generator.next()
for i in range(0,2):
    image = x[i]
    label = y[i]
    print (label)
    plt.imshow(image)
    plt.show()
'''
