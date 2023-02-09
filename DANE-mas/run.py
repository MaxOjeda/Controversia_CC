 model_config = {
        'net_shape': [200, 100],
        'att_shape': [200, 100],
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'is_init': True,
        'pretrain_params_path': './Log/cora/pretrain_params.pkl'
    }

trainer_config = {
        'net_shape': [200, 100],
        'att_shape': [200, 100],
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'drop_prob': 0.2,
        'learning_rate': 1e-5,
        'batch_size': 100,
        'num_epochs': 500,
        'beta': 100,
        'alpha': 50,
        'gamma': 500,
        'model_path': './Log/cora/cora_model.pkl',
    }


model = Model(model_config)
    trainer = Trainer(model, trainer_config)
    trainer.train(graph)
    trainer.infer(graph)