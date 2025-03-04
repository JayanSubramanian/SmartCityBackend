from preprocess import read_files, preprocess, GAT_pre_process_for_testing
import torch
from torch_geometric.data import Data
from final_code import GATWithDimensionalityReduction, in_channels, hidden_channels, out_channels, reduce_dim, num_heads

async def run_model(test_data):
    trans_data_test, prote_data_test, geno_data_test = await readfiles(test_data)
    X_test1,all_data_cleaned1 = preprocess(trans_data_test,prote_data_test,geno_data_test)

    edge_index, edge_attributes = GAT_pre_process_for_testing(all_data_cleaned1)
    edge_weights = edge_attributes
    edge_index = torch.tensor(edge_index)
    test_data = Data(x=X_test1, edge_index=edge_index, edge_attr=edge_weights)

    model1 = GATWithDimensionalityReduction(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, reduce_dim=reduce_dim, num_heads=num_heads)

    model1.load_state_dict(torch.load("GATWithDimensionalityReduction.pth",weights_only=True))

    model1.eval()
    with torch.no_grad():
        test_out = model1(test_data.x, test_data.edge_index, test_data.edge_attr)
        test_out = test_out.view(-1)

        probabilities = torch.sigmoid(test_out)

        average_probability = probabilities.mean()

        percentage_chance = average_probability.item()
    
        return percentage_chance