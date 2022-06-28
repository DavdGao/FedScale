import torch
import logging
import math
from fedscale.core.utils.nlp import mask_tokens
from torch.autograd import Variable
from fedscale.core.optimizer import ClientOptimizer

class Client(object):
    """Basic client component in Federated Learning"""
    def __init__(self, conf):
        self.optimizer = ClientOptimizer()
        pass

    def train(self, client_data, model, conf):

        clientId = conf.clientId
        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        tokenizer, device = conf.tokenizer, conf.device

        model = model.to(device=device)
        model.train()

        global_model = None

        assert conf.learning_rate == 1.0

        optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate)
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=device)

        epoch_train_loss = 1e-4

        error_type = None
        completed_steps = 0
        loss_squre = 0

        trained_samples = 0

        # TODO: One may hope to run fixed number of epochs, instead of iterations
        while completed_steps < conf.local_steps:
            try:
                for data_pair in client_data:
                    (data, target) = data_pair

                    data = Variable(data).to(device=device)
                    target = Variable(target).to(device=device)

                    output = model(data)
                    loss = criterion(output, target)

                    loss_list = loss.tolist()
                    loss = loss.mean()

                    temp_loss = sum(loss_list)/float(len(loss_list))
                    loss_squre = sum([l**2 for l in loss_list])/float(len(loss_list))
                    # only measure the loss of the first epoch
                    if completed_steps < len(client_data):
                        if epoch_train_loss == 1e-4:
                            epoch_train_loss = temp_loss
                        else:
                            epoch_train_loss = (1. - conf.loss_decay) * epoch_train_loss + conf.loss_decay * temp_loss

                    # ========= Define the backward loss ==============
                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                    optimizer.step()

                    # ========= Weight handler ========================
                    self.optimizer.update_client_weight(conf, model, global_model if global_model is not None else None  )

                    trained_samples += len(target)

                    completed_steps += 1

            except Exception as ex:
                error_type = ex
                break

            # after one epoch
            # completed_steps += 1

        state_dicts = model.state_dict()
        model_param = {p:state_dicts[p].data.cpu().numpy() for p in state_dicts}
        results = {
            'clientId':clientId,
            'moving_loss': epoch_train_loss,
            'trained_size': trained_samples,
            'success': completed_steps > 0
        }
        results['utility'] = math.sqrt(loss_squre)*float(trained_samples)

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes with {trained_samples} samples, {results}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        results['update_weight'] = model_param
        results['wall_duration'] = 0

        return results


    def test(self, conf):
        pass


