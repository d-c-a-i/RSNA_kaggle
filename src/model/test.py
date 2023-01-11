if __name__ == "__main__":
    import timm
    from torchsummary import summary
    model = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0)
    print(summary(model.cuda(), (3, 224, 224)))