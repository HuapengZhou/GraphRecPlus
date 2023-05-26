torch.save(graphrec.state_dict(), 'ModelSets/testModel.pth')

graphrec.load_state_dict(torch.load('ModelSets/testModel.pth'))
print("load the model successfully")
expected_rmse, mae = test(graphrec, device, test_loader)
print("expected_rmse" + str(expected_rmse))