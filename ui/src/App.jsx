import React from 'react';
import Box from '@material-ui/core/Box'
import Container from '@material-ui/core/Container'
import Grid from '@material-ui/core/Grid'
import { styled, ThemeProvider } from '@material-ui/core/styles'
import 'typeface-roboto'

import ServiceList from './ServiceList'
import logo from './assets/charade.png'
import theme from './theme'

const Logo = styled(Box)({
  height: '80px'
})

const Header = styled(Grid)({
  borderBottom: '1px solid grey',
  marginBottom: '20px'
})

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Container>

        <Header item xs={12}>
          <Logo component='img' src={logo} />
        </Header>

        <ServiceList />
      </Container>
    </ThemeProvider>
  )
}

export default App