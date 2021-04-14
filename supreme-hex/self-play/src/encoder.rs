use crate::{
    hex::{Hex, HexRepr, Player, Position},
    runtime::BatchItem,
};

pub struct EncodedState {
    // The shape of the encoded state.
    pub shape: Vec<u64>,
    // The encoded data
    pub data: Vec<f32>,
    // We could also embed a PhantomData here if we would like to track what encoder produced an EncodedState
}

pub trait Encoder: HexRepr {
    fn shape(size: usize) -> Vec<u64>;
    fn new(backing: BatchItem) -> Self;
    /// Convenience method to extract the encoding from the encoder. For use when e.g. serializing.
    /// Assumes that the encoder is commutative w.r.t. order of moves.
    /// In other words, it requires that `encoder.place(pos_a, player_a); encoder.place(pos_b, player_b); encoder.finalize() == encoder.place(pos_b, player_b); encoder.place(pos_a, player_a); encoder.finalize()`
    fn encoded(state: &Hex<()>) -> EncodedState
    where
        Self: Sized,
    {
        let shape = <Self as Encoder>::shape(state.size);
        let mut data = vec![0.0; shape.iter().product::<u64>() as usize];

        let mut encoder = Self::new(BatchItem {
            dims: shape.clone(),
            data: data.as_mut_ptr_range(),
        });

        // All the moves of the game
        for x in 0..state.size {
            for y in 0..state.size {
                for &player in &[Player::One, Player::Two] {
                    let position = Position { x, y };
                    if state.is_occupied(position, player) {
                        encoder.place(player, position);
                    }
                }
            }
        }

        // Force any deferred calculations
        encoder.finalize();

        // Return the shape and data.
        // At this point, the data will have been modified by the encoder.
        EncodedState { shape, data }
    }
}

#[derive(Debug)]
pub struct Normalized {
    pub inner: BatchItem,
    pub moves: [Vec<Position>; 2],
}

impl HexRepr for Normalized {
    fn place(&mut self, player: Player, position: Position) {
        let index = match player {
            Player::One => 0,
            Player::Two => 1,
        };

        self.moves[index].push(position);
    }

    fn unplace(&mut self, player: Player, position: Position) {
        let index = match player {
            Player::One => 0,
            Player::Two => 1,
        };
        // To ensure that it is called correctly.
        assert!(self.moves[index].pop() == Some(position));
    }

    fn finalize(&mut self) {
        // The current that will play next.
        let current_player = match self.moves[0].len() == self.moves[1].len() {
            true => Player::One,
            false => Player::Two,
        };

        // This encoder rotates the board such that the current-to-play always plays in the SW-NE direction, while the opponent plays in the NW-SE direction
        // with the following 'orentation'
        //
        //     NW     O     NE  <- (x, y) = (0, 0)
        //          O - O        <- (x, y) = (1, 0)
        //        O - O - O       <- (x, y) = (2, 0)
        //          O - O
        //     SW     O     SE
        // Since the hex board operates with Player 1 in the SE-NE direction and player 2 in the NW-SE direction, we will have to perform a rotation if it is player 2's turn

        // Zero the buffer
        self.inner.zero();

        match current_player {
            Player::One => {
                for plane in 0..=1 {
                    for position in &self.moves[plane] {
                        self.inner
                            .set(&[position.y as u64, position.x as u64, plane as u64], 1.0)
                    }
                }
            }
            Player::Two => {
                for plane in 0..=1 {
                    for position in &self.moves[1 - plane] {
                        self.inner
                            .set(&[position.x as u64, position.y as u64, plane as u64], 1.0);
                    }
                }
            }
        }

        let count = self.inner.dims.iter().product::<u64>() as usize;
        // We are responsible for providing the shape, so
        let size = *self.inner.dims.first().unwrap();
        let mut inner = vec![0.0; count];
        for x in 0..size {
            for y in 0..size {
                for z in 0..2 {
                    inner.push(self.inner.get(&[x, y, z]));
                }
            }
        }
    }
}

impl Encoder for Normalized {
    fn shape(size: usize) -> Vec<u64> {
        let size = size as u64;
        vec![size, size, 2]
    }

    fn new(backing: BatchItem) -> Self {
        Normalized {
            inner: backing,
            moves: [Vec::new(), Vec::new()],
        }
    }
}
