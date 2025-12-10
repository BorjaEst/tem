class LegacyModel(TEMModel):
    """Legacy compatibility wrapper for TEMModel.

    This class provides a drop-in replacement for the original `model.Model`
    class, adapting the new `TEMModel` interface to match the legacy API.

    Note:
        This class is intended
        for backward compatibility with existing codebases using the
        original `model.Model`. New code should use `TEMModel` directly.
    """

    def set_batch_size(self, batch_size: int):
        """Update batch size for memory storage.

        Args:
            batch_size: New batch size.
        """
        # Re-initialize storage with new batch size
        # We need to keep the same params and mask
        p_update_mask = self.storage.p_update_mask
        self.storage = MemoryStorage(self.config, p_update_mask, batch_size)

    def forward(self, walk, prev_iter=None, prev_M=None) -> List[TEMState]:
        """Forward pass for compatibility with legacy Model.

        Parameters
        ----------
        walk : List
            List of (locations, observations, actions) tuples.
        prev_iter : List[TEMState], optional
            List of states from previous run. If provided, the last state
            is used for initialization.
        prev_M : List[Tensor], optional
            Previous memory state [M_gen, M_inf]. Used if prev_iter is None.

        Returns
        -------
        List[TEMState]
            List of states for each step in the walk.
        """
        # Determine initial memory
        memory = None
        if prev_iter is not None and len(prev_iter) > 0:
            memory = prev_iter[-1].memory
        elif prev_M is not None:
            memory = prev_M

        # If memory is still None, create zero memory
        if memory is None:
            # Get batch size and device from first observation
            # walk[0] is (locations, x, a)
            x_0 = walk[0][1]
            batch_size = x_0.shape[0]
            device = x_0.device
            n_p_total = sum(self.config.n_p)

            # Create zero matrices
            M_gen = torch.zeros(batch_size, n_p_total, n_p_total, device=device)

            # Create M_inf if needed (dual memory)
            use_dual_memory = self.config.common_memory
            M_inf = torch.zeros(batch_size, n_p_total, n_p_total, device=device) if use_dual_memory else None

            memory = [M_gen, M_inf]

        # Run simulation
        # Note: Simulation is defined below, but available at runtime
        simulation = Simulation(self, walk, memory)

        # Collect all states
        steps = list(simulation)

        return steps
