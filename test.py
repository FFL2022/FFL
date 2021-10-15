from codeflaws.dataloader_cfl import CodeflawsCFLDGLStatementDataset

if __name__ == '__main__':
    dataset = CodeflawsCFLDGLStatementDataset()
    print(len(dataset))